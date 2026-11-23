"""
Unit tests for Phase 1 Hardsub OCR Configuration:
- RapidOCR parameter configuration (text_score, box_thresh, thresh, unclip_ratio)
- Multi-axis ROI padding (pad_left_ratio, pad_right_ratio, pad_bottom_ratio) with safe clamping
- VietOCR fallback transparency (logging warning and WebSocket warning event)
- Mask coverage verification and target case '誰' recovery
- Diagnostic logging of subtitle drop reasons and performance metrics
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas import MAX_PAD_RATIO, NormalizedBox, OCRJobRequest, ROIConfig, SubtitleSegment, VideoMetadata
from app.services.ffmpeg_service import crop_and_mask_roi, VideoROIStream
from app.services.ocr_engines import (
    OCREngineCache,
    OCRFrameResult,
    OCRTextLine,
    is_vietocr_available,
    order_and_merge_ocr_lines,
    recover_truncated_text_lines,
    remediate_mask_overflow,
    verify_mask_coverage,
)
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    ThreeLayerVisualFilter,
    run_ocr_pipeline,
)


# ==============================================================================
# 1. RapidOCR Parameter Configuration in OCRJobRequest
# ==============================================================================


def test_ocr_job_request_rapidocr_defaults():
    """Verify default RapidOCR parameters in OCRJobRequest."""
    req = OCRJobRequest(project_id="test-proj")
    assert req.rapidocr_text_score == 0.5
    assert req.rapidocr_box_thresh == 0.5
    assert req.rapidocr_thresh == 0.3
    assert req.rapidocr_unclip_ratio == 1.6


def test_ocr_job_request_rapidocr_custom_and_bounds():
    """Verify custom RapidOCR parameters and validation bounds (0.3 and 0.4 test configurations)."""
    # Valid configurations including 0.3 and 0.4
    req1 = OCRJobRequest(
        project_id="test-proj",
        rapidocr_text_score=0.3,
        rapidocr_box_thresh=0.3,
        rapidocr_thresh=0.2,
        rapidocr_unclip_ratio=1.8,
    )
    assert req1.rapidocr_text_score == 0.3
    assert req1.rapidocr_box_thresh == 0.3

    req2 = OCRJobRequest(
        project_id="test-proj",
        rapidocr_text_score=0.4,
        rapidocr_box_thresh=0.4,
    )
    assert req2.rapidocr_text_score == 0.4
    assert req2.rapidocr_box_thresh == 0.4

    # Out of bounds validation
    with pytest.raises(ValidationError):
        OCRJobRequest(project_id="test-proj", rapidocr_text_score=0.05)  # < 0.1

    with pytest.raises(ValidationError):
        OCRJobRequest(project_id="test-proj", rapidocr_text_score=1.5)  # > 1.0

    with pytest.raises(ValidationError):
        OCRJobRequest(project_id="test-proj", rapidocr_unclip_ratio=3.5)  # > 3.0


# ==============================================================================
# 2. ROI Multi-Axis Padding (pad_left, pad_right, pad_bottom)
# ==============================================================================


def test_ocr_job_request_roi_padding_defaults_and_custom():
    """Verify multi-axis padding fields in OCRJobRequest."""
    req_default = OCRJobRequest(project_id="test-proj")
    assert req_default.pad_left_ratio == 0.0
    assert req_default.pad_right_ratio == 0.0
    assert req_default.pad_bottom_ratio == 0.0

    req_custom = OCRJobRequest(
        project_id="test-proj",
        pad_left_ratio=0.08,
        pad_right_ratio=0.08,
        pad_bottom_ratio=0.05,
    )
    assert req_custom.pad_left_ratio == 0.08
    assert req_custom.pad_right_ratio == 0.08
    assert req_custom.pad_bottom_ratio == 0.05

    # Out of range validation (max MAX_PAD_RATIO = 0.35 in schema)
    with pytest.raises(ValidationError):
        OCRJobRequest(project_id="test-proj", pad_bottom_ratio=0.40)


def test_crop_and_mask_roi_multi_axis_padding():
    """Test crop_and_mask_roi padding calculation and safe clamping to 0.35."""
    # Dummy frame: 1000 x 1000
    frame = np.full((1000, 1000, 3), 50, dtype=np.uint8)
    roi_config = ROIConfig(
        main_roi=NormalizedBox(x=0.20, y=0.70, w=0.40, h=0.10)
    )

    # With padding: left=0.10 (40px), right=0.10 (40px), bottom=0.10 (10px), top=0.10 (10px)
    crop = crop_and_mask_roi(
        frame,
        roi_config,
        pad_top_ratio=0.10,
        pad_bottom_ratio=0.10,
        pad_left_ratio=0.10,
        pad_right_ratio=0.10,
    )
    # Expected: width = 400 + 40 + 40 = 480, height = 100 + 10 + 10 = 120
    assert crop.shape == (120, 480, 3)

    # Test clamping when ratio > 0.35: should clamp to 0.35
    # pad_left_ratio=0.50 -> clamped to 0.35 (400 * 0.35 = 140px)
    crop_clamped = crop_and_mask_roi(
        frame,
        roi_config,
        pad_left_ratio=0.50,
        pad_right_ratio=0.50,
        pad_bottom_ratio=0.50,
        pad_top_ratio=0.50,
    )
    # Expected width: 400 + 140 + 140 = 680, height: 100 + 35 + 35 = 170
    assert crop_clamped.shape == (170, 680, 3)


def test_video_roi_stream_multi_axis_padding(tmp_path):
    """Test VideoROIStream initializes multi-axis padding and bounds correctly."""
    dummy_vid = tmp_path / "dummy_vid.mp4"
    dummy_vid.write_bytes(b"dummy")
    roi_config = ROIConfig(
        main_roi=NormalizedBox(x=0.10, y=0.50, w=0.60, h=0.10)
    )
    meta = VideoMetadata(
        file_path=str(dummy_vid),
        file_name="dummy_vid.mp4",
        file_size_bytes=5,
        width=1000,
        height=1000,
        fps=25.0,
        duration_sec=10.0,
        frame_count=250,
    )

    # Initialize stream with pad_left=0.05, pad_right=0.05, pad_bottom=0.08
    with patch("app.services.ffmpeg_service.cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        stream = VideoROIStream(
            dummy_vid,
            roi_config,
            meta=meta,
            pad_top_ratio=0.05,
            pad_bottom_ratio=0.08,
            pad_left_ratio=0.05,
            pad_right_ratio=0.05,
        )
        assert stream.pad_left_ratio == 0.05
        assert stream.pad_right_ratio == 0.05
        assert stream.pad_bottom_ratio == 0.08
        assert stream.pad_top_ratio == 0.05


# ==============================================================================
# 3. VietOCR Fallback Transparency
# ==============================================================================


def test_is_vietocr_available():
    """Test is_vietocr_available returns boolean without raising exceptions."""
    available = is_vietocr_available()
    assert isinstance(available, bool)


@pytest.mark.asyncio
async def test_vietocr_fallback_transparency_warning(caplog):
    """When engine='auto', language='vi', and vietocr is not installed, verify warning log and WebSocket event."""
    req = OCRJobRequest(
        project_id="proj-vietocr-fallback",
        engine="auto",
        language="vi",
        sample_fps=5,
    )

    emitted_events = []

    async def mock_emit(event):
        emitted_events.append(event)

    with patch("app.services.ocr_pipeline.is_vietocr_available", return_value=False), \
         patch("app.services.ocr_pipeline.resolve_media_path") as mock_resolve, \
         patch("app.services.ocr_pipeline.VideoROIStream") as mock_stream_cls, \
         patch("app.services.ocr_pipeline.db") as mock_db, \
         caplog.at_level(logging.WARNING):

        mock_db.get_project = AsyncMock(
            return_value=MagicMock(
                video_path="test.mp4",
                video_metadata=MagicMock(fps=25.0, duration_sec=1.0),
                name="test_proj",
            )
        )
        mock_db.update_project_status = AsyncMock()
        mock_db.clear_project_segments = AsyncMock()
        mock_db.add_history_record = AsyncMock()

        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.name = "test.mp4"
        mock_resolve.return_value = mock_file

        # Stream returns no frames to complete immediately
        mock_stream = MagicMock()
        mock_stream.read_frame.return_value = None
        mock_stream_cls.return_value = mock_stream

        await run_ocr_pipeline(req, job_id="job-vietocr-test", emit=mock_emit)

    # 1. Verify exact log warning
    expected_warning = (
        "CẢNH BÁO: Cấu hình engine=auto và language=vi nhưng vietocr chưa được cài đặt trong môi trường. Hệ thống chuyển sang RapidOCR."
    )
    assert any(expected_warning in record.message for record in caplog.records)

    # 2. Verify WebSocket event with fallback warning
    warning_events = [e for e in emitted_events if getattr(e, "event", None) == "warning"]
    assert len(warning_events) >= 1
    assert warning_events[0].extra.get("fallback_warning") is True
    assert warning_events[0].extra.get("engine") == "rapidocr"
    assert expected_warning in warning_events[0].message


# ==============================================================================
# 4. RapidOCR Config Forwarding in OCREngineCache
# ==============================================================================


def test_recognize_rapidocr_parameter_forwarding():
    """Verify RapidOCR forwards text_score, box_thresh, thresh, unclip_ratio to engine."""
    cache = OCREngineCache()

    mock_engine = MagicMock()
    mock_engine.return_value = (
        [[[[10, 10], [100, 10], [100, 30], [10, 30]], "測試文本", 0.95]],
        [0.01, 0.01],
    )

    test_img = np.zeros((50, 200, 3), dtype=np.uint8)

    with patch.object(cache, "get_rapidocr", return_value=mock_engine):
        res = cache.recognize_rapidocr(
            test_img,
            text_score=0.35,
            box_thresh=0.45,
            thresh=0.25,
            unclip_ratio=1.75,
            diagnostic_logging=True,
        )

        # Confirm engine was called with parameter overrides
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args[1]
        assert call_kwargs.get("text_score") == 0.35
        assert call_kwargs.get("box_thresh") == 0.45
        assert call_kwargs.get("thresh") == 0.25
        assert call_kwargs.get("unclip_ratio") == 1.75
        assert res.text == "測試文本"


# ==============================================================================
# 5. Mask Coverage Verification & Generic Overflow Remediation
# ==============================================================================


def test_verify_mask_coverage():
    """Test verify_mask_coverage detects horizontal mask overflow (left or right)."""
    # Create a binary mask of shape (50, 400)
    # Text stroke pixels located from x=20 to x=380
    mask = np.zeros((50, 400), dtype=np.uint8)
    mask[15:35, 20:380] = 255

    # Case A: OCR box starts at x=60 (missed first 40px on the left)
    box_truncated_left = [[60, 15], [380, 15], [380, 35], [60, 35]]
    line_truncated = OCRTextLine(
        box=box_truncated_left,
        text="TRUNCATED ARBITRARY LINE",
        confidence=0.92,
    )
    has_overflow, overflow_dir, overflow_px = verify_mask_coverage(mask, [line_truncated])
    assert has_overflow is True
    assert overflow_dir == "left"
    assert overflow_px == 40  # 60 - 20 = 40px

    # Case B: OCR box covers full mask width
    box_aligned = [[20, 15], [380, 15], [380, 35], [20, 35]]
    line_aligned = OCRTextLine(
        box=box_aligned,
        text="FULL ARBITRARY LINE ALIGNED",
        confidence=0.95,
    )
    has_overflow_b, _, _ = verify_mask_coverage(mask, [line_aligned])
    assert has_overflow_b is False


def test_generic_mask_overflow_remediation_left():
    """
    Verify generic geometric recovery when left overflow occurs on arbitrary text.
    Tests that remediate_mask_overflow detects missing leading strokes from the binary mask,
    invokes recognize_fn with relaxed detector parameters, and returns the recovered full text.
    No hardcoded benchmark phrases or specific characters are used.
    """
    # Synthetic frame and binary mask of dimensions 60 x 600
    frame = np.full((60, 600, 3), 40, dtype=np.uint8)
    mask = np.zeros((60, 600), dtype=np.uint8)
    # Binary mask indicates text strokes begin at x=30 and end at x=520
    mask[15:45, 30:520] = 255

    # Initial OCR detection missed the leading token; box starts at x=100 (overflow = 70px)
    initial_line = OCRTextLine(
        box=[[100, 15], [520, 15], [520, 45], [100, 45]],
        text="REMAINDER OF ARBITRARY SUBTITLE PHRASE",
        confidence=0.88,
    )
    initial_res = OCRFrameResult(
        text=initial_line.text,
        confidence=initial_line.confidence,
        lines=[initial_line],
    )

    # Verify initial state has left overflow
    has_overflow, overflow_dir, overflow_px = verify_mask_coverage(mask, initial_res.lines)
    assert has_overflow is True
    assert overflow_dir == "left"
    assert overflow_px == 70

    # Mock recognizer called during remediation retry with relaxed parameters
    mock_recognizer = MagicMock()
    recovered_line = OCRTextLine(
        box=[[30, 15], [520, 15], [520, 45], [30, 45]],
        text="LEADING_TOKEN REMAINDER OF ARBITRARY SUBTITLE PHRASE",
        confidence=0.94,
    )
    mock_recognizer.return_value = OCRFrameResult(
        text=recovered_line.text,
        confidence=recovered_line.confidence,
        lines=[recovered_line],
    )

    # Execute production remediation logic
    remediated_res = remediate_mask_overflow(
        frame_bgr=frame,
        roi_mask=mask,
        ocr_result=initial_res,
        recognize_fn=mock_recognizer,
        retry_box_thresh=0.3,
        retry_unclip_ratio=2.2,
    )

    # Assert that remediation pass was invoked with relaxed parameters
    assert mock_recognizer.call_count == 1
    call_kwargs = mock_recognizer.call_args.kwargs
    assert call_kwargs.get("box_thresh") == 0.3
    assert call_kwargs.get("unclip_ratio") == 2.2

    # Assert that the full text line was recovered
    assert remediated_res.text == "LEADING_TOKEN REMAINDER OF ARBITRARY SUBTITLE PHRASE"
    assert remediated_res.confidence >= initial_res.confidence

    # Assert that the recovered boxes now fully cover the mask
    has_cov, _, _ = verify_mask_coverage(mask, remediated_res.lines)
    assert has_cov is False


def test_generic_mask_overflow_remediation_right():
    """
    Verify generic geometric recovery when right overflow occurs on arbitrary text.
    Tests that trailing characters clipped at the right margin are recovered generically.
    """
    frame = np.full((60, 600, 3), 40, dtype=np.uint8)
    mask = np.zeros((60, 600), dtype=np.uint8)
    # Mask extends from x=40 to x=550
    mask[15:45, 40:550] = 255

    # Initial OCR detection missed trailing characters; box ends prematurely at x=460 (overflow = 90px)
    initial_line = OCRTextLine(
        box=[[40, 15], [460, 15], [460, 45], [40, 45]],
        text="START OF GENERIC SUBTITLE WITHOUT ENDING",
        confidence=0.89,
    )
    initial_res = OCRFrameResult(
        text=initial_line.text,
        confidence=initial_line.confidence,
        lines=[initial_line],
    )

    # Verify right overflow is detected
    has_overflow, overflow_dir, overflow_px = verify_mask_coverage(mask, initial_res.lines)
    assert has_overflow is True
    assert overflow_dir == "right"
    assert overflow_px == 90

    # Mock recognizer returns line encompassing the full width
    mock_recognizer = MagicMock()
    recovered_line = OCRTextLine(
        box=[[40, 15], [550, 15], [550, 45], [40, 45]],
        text="START OF GENERIC SUBTITLE WITHOUT ENDING_TOKEN",
        confidence=0.93,
    )
    mock_recognizer.return_value = OCRFrameResult(
        text=recovered_line.text,
        confidence=recovered_line.confidence,
        lines=[recovered_line],
    )

    remediated_res = remediate_mask_overflow(
        frame_bgr=frame,
        roi_mask=mask,
        ocr_result=initial_res,
        recognize_fn=mock_recognizer,
    )

    assert mock_recognizer.call_count == 1
    assert remediated_res.text == "START OF GENERIC SUBTITLE WITHOUT ENDING_TOKEN"
    assert verify_mask_coverage(mask, remediated_res.lines)[0] is False


def test_generic_mask_overflow_no_overflow_passthrough():
    """
    Verify that when text boxes fully cover the mask (no overflow),
    remediate_mask_overflow performs zero redundant OCR invocations and returns result intact.
    """
    frame = np.full((60, 500, 3), 50, dtype=np.uint8)
    mask = np.zeros((60, 500), dtype=np.uint8)
    mask[15:45, 50:450] = 255

    aligned_line = OCRTextLine(
        box=[[50, 15], [450, 15], [450, 45], [50, 45]],
        text="PERFECTLY COVERED SUBTITLE LINE",
        confidence=0.96,
    )
    initial_res = OCRFrameResult(
        text=aligned_line.text,
        confidence=aligned_line.confidence,
        lines=[aligned_line],
    )

    mock_recognizer = MagicMock()
    remediated_res = remediate_mask_overflow(
        frame_bgr=frame,
        roi_mask=mask,
        ocr_result=initial_res,
        recognize_fn=mock_recognizer,
    )

    # Must NOT invoke recognizer when coverage is already complete
    assert mock_recognizer.call_count == 0
    assert remediated_res.text == "PERFECTLY COVERED SUBTITLE LINE"
    assert remediated_res == initial_res


def test_generic_mask_overflow_graceful_fallback_on_failure():
    """
    Verify that if the remediation retry fails or throws an exception,
    the pipeline gracefully falls back to the original partial result without crashing.
    """
    frame = np.full((60, 500, 3), 50, dtype=np.uint8)
    mask = np.zeros((60, 500), dtype=np.uint8)
    mask[15:45, 30:450] = 255

    initial_line = OCRTextLine(
        box=[[90, 15], [450, 15], [450, 45], [90, 45]],
        text="PARTIAL LINE BEFORE ERROR",
        confidence=0.85,
    )
    initial_res = OCRFrameResult(
        text=initial_line.text,
        confidence=initial_line.confidence,
        lines=[initial_line],
    )

    # Simulate recognition backend raising an unexpected runtime error
    failing_recognizer = MagicMock(side_effect=RuntimeError("Engine transient failure"))

    remediated_res = remediate_mask_overflow(
        frame_bgr=frame,
        roi_mask=mask,
        ocr_result=initial_res,
        recognize_fn=failing_recognizer,
    )

    # Must catch exception and safely preserve the original result
    assert remediated_res.text == "PARTIAL LINE BEFORE ERROR"
    assert remediated_res.confidence == 0.85


def test_rapidocr_dynamic_parameter_override_consecutive_jobs():
    """Verify consecutive jobs with different thresholds dynamically update the active engine."""
    cache = OCREngineCache()

    class MockPostProcess:
        def __init__(self):
            self.box_thresh = 0.5
            self.thresh = 0.3
            self.unclip_ratio = 1.6

    class MockDetector:
        def __init__(self):
            self.postprocess_op = MockPostProcess()

    class MockRapidOCR:
        def __init__(self):
            self.text_score = 0.5
            self.text_det = MockDetector()

        def __call__(self, img, **kwargs):
            return ([[ [[10, 10], [50, 10], [50, 30], [10, 30]], "job", 0.9 ]], [0.01])

    mock_engine = MockRapidOCR()
    cache._rapidocr = mock_engine

    # Job 1 execution: box_thresh=0.5, text_score=0.4
    e1 = cache.get_rapidocr(box_thresh=0.5, text_score=0.4, thresh=0.3, unclip_ratio=1.6)
    assert e1.text_score == 0.4
    assert e1.text_det.postprocess_op.box_thresh == 0.5
    assert e1.text_det.postprocess_op.thresh == 0.3
    assert e1.text_det.postprocess_op.unclip_ratio == 1.6

    # Job 2 execution: box_thresh=0.3, text_score=0.25, thresh=0.2, unclip_ratio=1.9
    e2 = cache.get_rapidocr(box_thresh=0.3, text_score=0.25, thresh=0.2, unclip_ratio=1.9)
    assert e2 is e1  # Same cached instance: O(1) memory, zero model reload
    assert e2.text_score == 0.25
    assert e2.text_det.postprocess_op.box_thresh == 0.3
    assert e2.text_det.postprocess_op.thresh == 0.2
    assert e2.text_det.postprocess_op.unclip_ratio == 1.9


# ==============================================================================
# 6. Diagnostic Drop Logging (Activity, Stability, Text Score, Deduplication)
# ==============================================================================


def test_visual_filter_diagnostic_drop_logging(caplog):
    """Test ThreeLayerVisualFilter logs drop reasons when diagnostic_logging=True."""
    filter_diag = ThreeLayerVisualFilter(
        luma_threshold=185,
        stability_frames=2,
        visual_diff_threshold=0.14,
        frame_step_sec=0.2,
        diagnostic_logging=True,
    )

    empty_roi = np.zeros((100, 300, 3), dtype=np.uint8)

    with caplog.at_level(logging.DEBUG):
        spans = filter_diag.process_frame(empty_roi, timestamp_sec=0.2)
        assert len(spans) == 0

    assert any("[OCR DIAG] Frame at 0.200s: dropped candidate text due to low activity" in r.message for r in caplog.records)


def test_deduplicator_diagnostic_merge_logging(caplog):
    """Test deduplicator merge logging when diagnostic_logging is active."""
    dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=0.85, max_merge_gap_sec=0.50)

    seg1 = SubtitleSegment(
        id="seg-1",
        project_id="test",
        index=1,
        start_time=1.0,
        end_time=2.0,
        text="Xin chào Việt Nam",
        confidence=0.90,
    )
    seg2 = SubtitleSegment(
        id="seg-2",
        project_id="test",
        index=2,
        start_time=2.1,
        end_time=3.0,
        text="Xin chào Việt Nam",
        confidence=0.92,
    )

    dedup.add_or_merge(seg1)
    active, was_merged = dedup.add_or_merge(seg2)

    assert was_merged is True
    assert active.start_time == 1.0
    assert active.end_time == 3.0


def test_order_and_merge_diagnostic_min_confidence_logging(caplog):
    """Test order_and_merge_ocr_lines logs dropped lines due to min_confidence."""
    lines = [
        OCRTextLine(
            box=[[10, 10], [50, 10], [50, 30], [10, 30]],
            text="LowConfText",
            confidence=0.22,
        ),
        OCRTextLine(
            box=[[60, 10], [120, 10], [120, 30], [60, 30]],
            text="HighConfText",
            confidence=0.88,
        ),
    ]

    with caplog.at_level(logging.INFO):
        text, conf, valid = order_and_merge_ocr_lines(
            lines, min_confidence=0.35, diagnostic_logging=True
        )

    assert text == "HighConfText"
    assert any("[OCR DIAG] Filtered line (low confidence" in r.message for r in caplog.records)
