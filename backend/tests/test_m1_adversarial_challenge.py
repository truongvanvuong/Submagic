"""
Adversarial Stress & Edge-Case Verification Suite for Milestone 1:
- RapidOCR parameter configuration and boundary overrides
- Multi-axis ROI padding with extreme ratios and edge clamping
- Mask coverage verification on synthetic edge cases and threshold limits
- Vietnamese dấu nặng bottom diacritics preservation via pad_bottom_ratio
- VietOCR fallback transparency and warning event emission
- Target case 誰 recovery stress testing under real-world conditions
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas import JobStatus, NormalizedBox, OCRJobRequest, ROIConfig
from app.services.ffmpeg_service import crop_and_mask_roi, VideoROIStream
from app.services.ocr_engines import (
    MaskCoverageResult,
    OCREngineCache,
    OCRFrameResult,
    OCRTextLine,
    is_vietocr_available,
    recover_truncated_text_lines,
    verify_mask_coverage,
)
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    ThreeLayerVisualFilter,
    run_ocr_pipeline,
)


# ==============================================================================
# 1. RapidOCR Parameter Overrides & Boundary Stress
# ==============================================================================

class TestRapidOCRParameterStress:
    """Stress-test RapidOCR parameter validation and forwarding behavior."""

    @pytest.mark.parametrize("text_score", [0.1, 0.3, 0.4, 0.5, 1.0])
    def test_text_score_valid_boundaries(self, text_score: float):
        req = OCRJobRequest(project_id="test", rapidocr_text_score=text_score)
        assert req.rapidocr_text_score == pytest.approx(text_score)

    @pytest.mark.parametrize("invalid_score", [0.05, 0.09, 1.01, 1.5, -0.1])
    def test_text_score_invalid_boundaries(self, invalid_score: float):
        with pytest.raises(ValidationError):
            OCRJobRequest(project_id="test", rapidocr_text_score=invalid_score)

    @pytest.mark.parametrize("box_thresh", [0.1, 0.3, 0.4, 0.5, 1.0])
    def test_box_thresh_valid_boundaries(self, box_thresh: float):
        req = OCRJobRequest(project_id="test", rapidocr_box_thresh=box_thresh)
        assert req.rapidocr_box_thresh == pytest.approx(box_thresh)

    @pytest.mark.parametrize("invalid_box", [0.05, 1.05, -0.5])
    def test_box_thresh_invalid_boundaries(self, invalid_box: float):
        with pytest.raises(ValidationError):
            OCRJobRequest(project_id="test", rapidocr_box_thresh=invalid_box)

    @pytest.mark.parametrize("unclip", [1.0, 1.6, 2.0, 3.0])
    def test_unclip_ratio_valid_boundaries(self, unclip: float):
        req = OCRJobRequest(project_id="test", rapidocr_unclip_ratio=unclip)
        assert req.rapidocr_unclip_ratio == pytest.approx(unclip)

    @pytest.mark.parametrize("invalid_unclip", [0.9, 0.99, 3.01, 4.0])
    def test_unclip_ratio_invalid_boundaries(self, invalid_unclip: float):
        with pytest.raises(ValidationError):
            OCRJobRequest(project_id="test", rapidocr_unclip_ratio=invalid_unclip)

    def test_recognize_rapidocr_kwargs_forwarding_and_type_error_resilience(self):
        """Verify kwargs are forwarded to the engine, and TypeError falls back gracefully."""
        cache = OCREngineCache()

        # Mock engine that tracks received kwargs
        recorded_kwargs = {}

        def mock_engine(img, **kwargs):
            recorded_kwargs.update(kwargs)
            return ([[ [[10, 10], [50, 10], [50, 30], [10, 30]], "test", 0.9 ]], [0.01])

        cache._rapidocr = mock_engine

        test_img = np.ones((50, 100, 3), dtype=np.uint8) * 200
        res = cache.recognize_rapidocr(
            test_img,
            text_score=0.35,
            box_thresh=0.45,
            thresh=0.25,
            unclip_ratio=1.75,
        )

        assert res.text == "test"
        assert recorded_kwargs.get("text_score") == 0.35
        assert recorded_kwargs.get("box_thresh") == 0.45
        assert recorded_kwargs.get("thresh") == 0.25
        assert recorded_kwargs.get("unclip_ratio") == 1.75

        # Test engine that rejects kwargs (legacy RapidOCR call signature)
        def strict_engine(img):
            return ([[ [[10, 10], [50, 10], [50, 30], [10, 30]], "legacy", 0.88 ]], [0.01])

        cache._rapidocr = strict_engine
        res_legacy = cache.recognize_rapidocr(test_img, text_score=0.3)
        assert res_legacy.text == "legacy"


# ==============================================================================
# 2. Multi-Axis ROI Padding & Extreme Aspect Ratios
# ==============================================================================

class TestMultiAxisPaddingExtreme:
    """Stress-test multi-axis ROI padding with extreme ratios and edge boundaries."""

    def test_negative_and_excessive_padding_clamping(self):
        """Values < 0.0 clamp to 0.0; values > 0.35 clamp to 0.35."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_cfg = ROIConfig(
            main_roi=NormalizedBox(x=0.1, y=0.7, w=0.8, h=0.2)
        )
        base_h = int(0.2 * 720)   # 144
        base_w = int(0.8 * 1280)  # 1024

        # Unpadded
        cropped_base = crop_and_mask_roi(img, roi_cfg)
        assert cropped_base.shape == (base_h, base_w, 3)

        # Negative ratios should not shrink below base ROI
        cropped_neg = crop_and_mask_roi(
            img, roi_cfg,
            pad_top_ratio=-0.2, pad_bottom_ratio=-0.5,
            pad_left_ratio=-0.3, pad_right_ratio=-0.1,
        )
        assert cropped_neg.shape == (base_h, base_w, 3)

        # Excessive ratios (> 0.35) should be clamped strictly to 0.35
        cropped_excess = crop_and_mask_roi(
            img, roi_cfg,
            pad_top_ratio=0.80,
            pad_bottom_ratio=0.90,
            pad_left_ratio=0.75,
            pad_right_ratio=0.60,
        )
        expected_pad_t = int(base_h * 0.35)
        expected_pad_b = int(base_h * 0.35)
        expected_pad_l = int(base_w * 0.35)
        expected_pad_r = int(base_w * 0.35)

        expected_h = min(720, base_h + expected_pad_t + expected_pad_b)
        expected_w = min(1280, base_w + expected_pad_l + expected_pad_r)
        assert cropped_excess.shape == (expected_h, expected_w, 3)

    def test_boundary_frame_touching_edges(self):
        """ROI touching video frame boundaries must not cause negative indexing or overflow."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # ROI touching top-left corner
        roi_top_left = ROIConfig(main_roi=NormalizedBox(x=0.0, y=0.0, w=0.5, h=0.5))
        cropped = crop_and_mask_roi(
            img, roi_top_left,
            pad_top_ratio=0.35, pad_left_ratio=0.35,
        )
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

        # ROI touching bottom-right corner
        roi_bottom_right = ROIConfig(main_roi=NormalizedBox(x=0.5, y=0.5, w=0.5, h=0.5))
        cropped_br = crop_and_mask_roi(
            img, roi_bottom_right,
            pad_bottom_ratio=0.35, pad_right_ratio=0.35,
        )
        assert cropped_br.shape[0] > 0
        assert cropped_br.shape[1] > 0

    def test_video_roi_stream_even_dimensions_with_padding(self, tmp_path):
        """VideoROIStream must maintain even dimensions (rw & ~1, rh & ~1) and minimum 16px."""
        from app.schemas import VideoMetadata

        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        roi_cfg = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))
        meta = VideoMetadata(
            file_path=str(dummy_video),
            file_name="dummy.mp4",
            file_size_bytes=1000,
            duration_sec=5.0,
            fps=25.0,
            frame_count=125,
            width=1280,
            height=720,
        )

        with patch("cv2.VideoCapture") as mock_cv:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv.return_value = mock_cap

            stream = VideoROIStream(
                dummy_video,
                roi_cfg,
                meta=meta,
                pad_top_ratio=0.15,
                pad_bottom_ratio=0.12,
                pad_left_ratio=0.08,
                pad_right_ratio=0.08,
            )

            # Both rw and rh must be even numbers for FFmpeg rawvideo codecs
            assert stream.rw % 2 == 0
            assert stream.rh % 2 == 0
            assert stream.rw >= 16
            assert stream.rh >= 16


# ==============================================================================
# 3. Mask Coverage Verification Edge Cases
# ==============================================================================

class TestMaskCoverageVerificationAdversarial:
    """Stress-test verify_mask_coverage on malformed, edge-case, and boundary inputs."""

    def test_degenerate_and_empty_inputs(self):
        """verify_mask_coverage must gracefully return (False, None, 0) without exceptions."""
        # None mask
        assert verify_mask_coverage(None, [OCRTextLine(text="a", box=[[0, 0], [10, 0], [10, 10], [0, 10]])]) == (False, None, 0)

        # Empty ndarray
        assert verify_mask_coverage(np.array([], dtype=np.uint8), []) == (False, None, 0)

        # All-zero mask (no foreground pixels found)
        zero_mask = np.zeros((50, 100), dtype=np.uint8)
        assert verify_mask_coverage(zero_mask, [OCRTextLine(text="a", box=[[0, 0], [10, 0], [10, 10], [0, 10]])]) == (False, None, 0)

        # Empty lines list
        mask = np.ones((50, 100), dtype=np.uint8) * 255
        assert verify_mask_coverage(mask, []) == (False, None, 0)

        # Lines without bounding box
        line_no_box = OCRTextLine(text="hello", box=None)
        assert verify_mask_coverage(mask, [line_no_box]) == (False, None, 0)

        # Lines with malformed box (< 4 points)
        line_bad_box = OCRTextLine(text="hello", box=[[0, 0], [10, 10]])
        assert verify_mask_coverage(mask, [line_bad_box]) == (False, None, 0)

    def test_right_overflow_detection(self):
        """Mask extending significantly to the right of OCR box must return (True, 'right', px)."""
        mask = np.zeros((50, 500), dtype=np.uint8)
        mask[15:35, 100:400] = 255  # mx2 = 400

        # OCR box ends at 360 (40px trailing overflow)
        lines = [
            OCRTextLine(
                box=[[100, 15], [360, 15], [360, 35], [100, 35]],
                text="Some text truncated on right",
                confidence=0.9,
            )
        ]
        has_overflow, overflow_dir, overflow_px = verify_mask_coverage(mask, lines)
        assert has_overflow is True
        assert overflow_dir == "right"
        assert overflow_px == 40

    def test_balanced_mask_no_overflow(self):
        """When mask is contained within OCR bounding boxes, returns (False, None, 0)."""
        mask = np.zeros((50, 500), dtype=np.uint8)
        mask[15:35, 110:390] = 255  # Inside 100..400

        lines = [
            OCRTextLine(
                box=[[100, 15], [400, 15], [400, 35], [100, 35]],
                text="Centered subtitle text",
                confidence=0.95,
            )
        ]
        has_overflow, overflow_dir, overflow_px = verify_mask_coverage(mask, lines)
        assert has_overflow is False
        assert overflow_dir is None
        assert overflow_px == 0

    def test_empirical_finding_8px_truncation_missed_by_default_threshold(self):
        """
        EMPIRICAL FINDING:
        With default min_overflow_ratio=0.85, an 8px truncation on a 24px line height
        evaluates to 8 < (24 * 0.85 = 20.4px), returning has_overflow = False!
        This documents the exact empirical behavior of verify_mask_coverage on small truncations.
        """
        mask = np.zeros((50, 500), dtype=np.uint8)
        mask[13:37, 368:600] = 255  # mh = 24, mx1 = 368

        # 8px truncation (box starts at 376 instead of 368)
        lines = [
            OCRTextLine(
                box=[[376, 13], [600, 13], [600, 37], [376, 37]],
                text="能想到東方不敗也敗在別人手下了",
                confidence=0.91,
            )
        ]

        # Under default min_overflow_ratio=0.85: thresh = 20.4px -> 8px overflow is NOT flagged
        has_overflow_default, _, _ = verify_mask_coverage(mask, lines, min_overflow_ratio=0.85)
        assert has_overflow_default is False

        # Under sensitive threshold (e.g. min_overflow_ratio=0.30): thresh = 7.2px -> flagged!
        has_overflow_sens, overflow_dir, overflow_px = verify_mask_coverage(mask, lines, min_overflow_ratio=0.30)
        assert has_overflow_sens is True
        assert overflow_dir == "left"
        assert overflow_px == 8


# ==============================================================================
# 4. Vietnamese Dấu Nặng (Bottom Diacritic) Preservation
# ==============================================================================

class TestVietnameseDauNangPreservation:
    """Empirically test that pad_bottom_ratio prevents clipping bottom diacritics (dấu nặng)."""

    def test_dau_nang_dot_preservation_with_pad_bottom(self):
        """
        Synthetic Vietnamese subtitle line:
        Base character occupies y=580..645.
        The dấu nặng dot (ạ, ặ, ệ, ị, ọ, ộ, ợ, ụ, ự, ỵ) is placed at y=647..650.
        ROI is configured tightly with bottom at y=646.
        - With pad_bottom_ratio=0.0: dot is sliced off.
        - With pad_bottom_ratio=0.08: dot is fully preserved in cropped ROI.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw base character stroke
        frame[580:645, 400:600] = 255
        # Draw dấu nặng dot below baseline (y=647..650)
        frame[647:651, 495:505] = 255

        # ROI with bottom boundary at y=646 (y=576 to y=648, h=72)
        roi_cfg = ROIConfig(
            main_roi=NormalizedBox(x=0.25, y=0.80, w=0.50, h=0.10)  # y1=576, y2=648
        )

        # 1. Without bottom padding (pad_bottom_ratio=0.0)
        tight_roi = ROIConfig(main_roi=NormalizedBox(x=0.25, y=580/720, w=0.50, h=65/720)) # ry2 = 645
        crop_tight = crop_and_mask_roi(frame, tight_roi, pad_bottom_ratio=0.0)
        # Crop height is 65 (y=580 to y=645). Frame y=647..651 (dấu nặng dot) is strictly excluded!
        assert crop_tight.shape[0] == 65
        # The dot pixels at y=647..651 are completely absent from unpadded crop
        assert crop_tight.shape[0] < (651 - 580)

        # 2. With bottom padding (pad_bottom_ratio=0.10):
        # roi_h = 65, pad_b = int(65 * 0.10) = 6 -> ry2 expands from 645 to 651!
        crop_padded = crop_and_mask_roi(frame, tight_roi, pad_bottom_ratio=0.10)
        assert crop_padded.shape[0] == 71  # 65 + 6
        # The dot rows (y=647..650) are now inside the padded crop (rows 67..70 relative to ry1=580)
        dot_region_in_crop = crop_padded[67:71, :]
        assert np.any(dot_region_in_crop == 255), "Dấu nặng dot must be preserved by pad_bottom_ratio"


# ==============================================================================
# 5. VietOCR Fallback Transparency & Warning Event
# ==============================================================================

class TestVietOCRFallbackTransparency:
    """Empirically test VietOCR fallback logging and WebSocket warning events."""

    @pytest.mark.asyncio
    async def test_fallback_warning_when_vietocr_missing(self, tmp_path, caplog):
        """When engine='auto', language='vi', and vietocr is uninstalled, warning event is fired."""
        dummy_video = tmp_path / "sample.mp4"
        dummy_video.write_bytes(b"dummy_video_content")

        mock_project = MagicMock(
            video_path=str(dummy_video),
            video_metadata=MagicMock(fps=25.0, duration_sec=1.0),
            name="test_proj",
        )

        events_emitted = []

        async def capture_event(ev):
            events_emitted.append(ev)

        req = OCRJobRequest(
            project_id="proj-vi-fallback",
            language="vi",
            engine="auto",
        )

        with patch("app.services.ocr_pipeline.is_vietocr_available", return_value=False), \
             patch("app.services.ocr_pipeline.resolve_media_path", return_value=dummy_video), \
             patch("app.services.ocr_pipeline.VideoROIStream") as mock_stream_cls, \
             patch("app.services.ocr_pipeline.db") as mock_db, \
             caplog.at_level(logging.WARNING):

            mock_db.get_project = AsyncMock(return_value=mock_project)
            mock_db.update_project_status = AsyncMock()
            mock_db.clear_project_segments = AsyncMock()
            mock_db.add_history_record = AsyncMock()

            mock_stream = MagicMock()
            mock_stream.read_frame.return_value = None
            mock_stream_cls.return_value = mock_stream

            await run_ocr_pipeline(
                req=req,
                job_id="job-fallback-test",
                emit=capture_event,
            )

        # 1. Assert log warning was generated
        assert any("vietocr chưa được cài đặt" in r.message for r in caplog.records)

        # 2. Assert WebSocket warning event was emitted
        warning_events = [ev for ev in events_emitted if getattr(ev, "event", None) == "warning"]
        assert len(warning_events) >= 1
        assert warning_events[0].extra.get("fallback_warning") is True
        assert warning_events[0].extra.get("engine") == "rapidocr"

    @pytest.mark.asyncio
    async def test_no_warning_when_language_is_not_vi(self, tmp_path, caplog):
        """When language='en', no fallback warning should be emitted."""
        dummy_video = tmp_path / "sample_en.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_project = MagicMock(
            video_path=str(dummy_video),
            video_metadata=MagicMock(fps=25.0, duration_sec=1.0),
            name="test_proj",
        )

        events_emitted = []

        async def capture_event(ev):
            events_emitted.append(ev)

        req = OCRJobRequest(
            project_id="proj-en",
            language="en",
            engine="auto",
        )

        with patch("app.services.ocr_pipeline.is_vietocr_available", return_value=False), \
             patch("app.services.ocr_pipeline.resolve_media_path", return_value=dummy_video), \
             patch("app.services.ocr_pipeline.VideoROIStream") as mock_stream_cls, \
             patch("app.services.ocr_pipeline.db") as mock_db, \
             caplog.at_level(logging.WARNING):

            mock_db.get_project = AsyncMock(return_value=mock_project)
            mock_db.update_project_status = AsyncMock()
            mock_db.clear_project_segments = AsyncMock()
            mock_db.add_history_record = AsyncMock()

            mock_stream = MagicMock()
            mock_stream.read_frame.return_value = None
            mock_stream_cls.return_value = mock_stream

            await run_ocr_pipeline(
                req=req,
                job_id="job-en-test",
                emit=capture_event,
            )

        warning_events = [ev for ev in events_emitted if getattr(ev, "event", None) == "warning"]
        assert len(warning_events) == 0


# ==============================================================================
# 6. Target Case 誰 Recovery Empirical Challenge
# ==============================================================================

class TestTargetCaseSheRecoveryChallenge:
    """Stress-test the target case '誰' recovery under various edge scenarios."""

    def test_empirical_she_recovery_logic_gap_with_best_mask_present(self):
        """
        EMPIRICAL CHALLENGE:
        In ocr_pipeline.py lines 1128-1144:
            if span.best_mask is not None and ocr_res.lines:
                has_overflow, overflow_dir, overflow_px = verify_mask_coverage(...)
                if has_overflow and overflow_dir == 'left':
                    if '能想到東方不敗' in ocr_res.text ...:
                        ocr_res.text = '誰' + ...
            elif '能想到東方不敗' in ocr_res.text ...:
                ocr_res.text = '誰' + ...

        If span.best_mask is present, but has_overflow is False (e.g. 8px truncation
        which does not exceed the default 20px threshold, or text recognized as simplified 谁
        with an aligned box), the elif is SKIPPED, and 誰 is NOT restored!
        """
        mask = np.zeros((50, 500), dtype=np.uint8)
        mask[13:37, 368:600] = 255

        # 8px truncation box
        lines = [
            OCRTextLine(
                box=[[376, 13], [600, 13], [600, 37], [376, 37]],
                text="能想到東方不敗也敗在別人手下了",
                confidence=0.91,
            )
        ]

        text = lines[0].text
        best_mask = mask

        # Trace exact pipeline logic from ocr_pipeline.py:
        if best_mask is not None and lines:
            has_overflow, overflow_dir, overflow_px = verify_mask_coverage(
                best_mask, lines
            )
            if has_overflow and overflow_dir == "left":
                if "能想到東方不敗" in text and not text.startswith("誰"):
                    text = "誰" + (text[1:] if text.startswith("谁") else text)
        elif "能想到東方不敗" in text and not text.startswith("誰"):
            text = "誰" + (text[1:] if text.startswith("谁") else text)

        # Under the current code, has_overflow is False, so text is NOT recovered!
        is_recovered = text.startswith("誰")
        # Document the empirical reality:
        assert is_recovered is False, (
            "Empirical verification shows that current ocr_pipeline.py fails to recover "
            "誰 when best_mask is present with an 8px truncation due to verify_mask_coverage threshold."
        )

    def test_ideal_robust_recovery_catches_all_she_cases(self):
        """Verify that an unconditional check reliably restores 誰 for all candidate variants."""
        variants = [
            "能想到東方不敗也敗在別人手下了",
            "谁能想到東方不敗也敗在別人手下了",
            "誰能想到東方不敗也敗在別人手下了",  # Already has traditional 誰
        ]

        for var in variants:
            text = var
            if "能想到東方不敗" in text and not text.startswith("誰"):
                text = "誰" + (text[1:] if text.startswith("谁") else text)

            assert text == "誰能想到東方不敗也敗在別人手下了"
            assert text.count("誰") == 1


# ==============================================================================
# 7. Milestone 1 Iteration 2: Adversarial Truncation Matrix (4, 8, 15, 30, 60 px)
# ==============================================================================

class TestAdversarialTruncationMatrix:
    """
    Stress-test verify_mask_coverage and recover_truncated_text_lines across
    truncation boundaries (4px, 8px, 15px, 30px, 60px) and directions
    ('left', 'right', 'both', 'neither').
    """

    @pytest.mark.parametrize("truncation_px", [4, 8, 15, 30, 60])
    @pytest.mark.parametrize("direction", ["left", "right", "both", "neither"])
    def test_truncation_matrix_verify_and_recovery(self, truncation_px: int, direction: str):
        line_h = 30
        w = 1000
        h = 60
        base_ox1 = 200
        base_ox2 = 600

        box = [[base_ox1, 15], [base_ox2, 15], [base_ox2, 15 + line_h], [base_ox1, 15 + line_h]]
        lines = [OCRTextLine(text="ARBITRARY_BASE_LINE", box=box, confidence=0.85)]
        ocr_res = OCRFrameResult(text="ARBITRARY_BASE_LINE", confidence=0.85, lines=lines)
        dummy_img = np.full((h, w, 3), 128, dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        if direction == "left":
            mx1 = base_ox1 - truncation_px
            mx2 = base_ox2
        elif direction == "right":
            mx1 = base_ox1
            mx2 = base_ox2 + truncation_px
        elif direction == "both":
            mx1 = base_ox1 - truncation_px
            mx2 = base_ox2 + truncation_px
        else:  # neither
            mx1 = base_ox1 + truncation_px
            mx2 = base_ox2 - truncation_px

        mask[15:15 + line_h, max(0, mx1):min(w, mx2)] = 255

        # 1. verify_mask_coverage with default threshold (min_overflow_ratio=0.85 -> thresh=25.5px)
        cov_default = verify_mask_coverage(mask, lines, min_overflow_ratio=0.85)
        if direction == "neither" or truncation_px < 25.5:
            assert cov_default.has_overflow is False
            assert cov_default.overflow_dir is None
            assert cov_default.overflow_px == 0
        else:
            assert cov_default.has_overflow is True
            assert cov_default.overflow_dir == direction
            assert cov_default.overflow_px == truncation_px

        # 2. verify_mask_coverage with sensitive thresh_px=3.0 detects small truncations (4, 8, 15)
        cov_sens = verify_mask_coverage(mask, lines, thresh_px=3.0)
        if direction == "neither":
            assert cov_sens.has_overflow is False
            assert cov_sens.overflow_dir is None
            assert cov_sens.overflow_px == 0
        else:
            assert cov_sens.has_overflow is True
            assert cov_sens.overflow_dir == direction
            assert cov_sens.overflow_px == truncation_px

        # 3. Test recover_truncated_text_lines behavior
        re_ocr_calls = []

        def mock_recognize(strip, **kw):
            re_ocr_calls.append((strip.shape, kw))
            # Return expanded text with higher confidence
            return OCRFrameResult(
                text="EXTENDED_ARBITRARY_RECOVERED_LINE",
                confidence=0.95,
                lines=[OCRTextLine(
                    text="EXTENDED_ARBITRARY_RECOVERED_LINE",
                    box=[[0, 0], [strip.shape[1], 0], [strip.shape[1], strip.shape[0]], [0, strip.shape[0]]],
                    confidence=0.95,
                )],
            )

        # recover uses min_overflow_ratio=0.75 by default (thresh = 30 * 0.75 = 22.5px)
        recovered = recover_truncated_text_lines(dummy_img, mask, ocr_res, recognize_fn=mock_recognize)

        if direction != "neither" and truncation_px >= 22.5:
            # Must trigger targeted re-OCR strip
            assert len(re_ocr_calls) == 1
            crop_h, crop_w = re_ocr_calls[0][0][:2]
            assert crop_h >= line_h
            if direction == "left":
                # crop_x1 should be expanded to include mx1
                assert crop_w > (base_ox2 - base_ox1)
            elif direction == "right":
                assert crop_w > (base_ox2 - base_ox1)
            elif direction == "both":
                assert crop_w >= (base_ox2 - base_ox1) + 2 * truncation_px
            assert recovered.text == "EXTENDED_ARBITRARY_RECOVERED_LINE"
            assert recovered.confidence == 0.95
        else:
            # Sub-threshold or contained: zero re-OCR calls, returns original result intact
            assert len(re_ocr_calls) == 0
            assert recovered == ocr_res


# ==============================================================================
# 8. Milestone 1 Iteration 2: Abnormal Mask Resilience Stress
# ==============================================================================

class TestAbnormalMaskResilience:
    """Stress-test zero masks, full-white masks, noise specks, and watermarks."""

    def test_zero_mask_resilience(self):
        """All-zero masks must never crash and must never trigger re-OCR."""
        zero_mask = np.zeros((60, 400), dtype=np.uint8)
        lines = [OCRTextLine(text="test", box=[[50, 15], [200, 15], [200, 45], [50, 45]])]
        ocr_res = OCRFrameResult(text="test", confidence=0.9, lines=lines)
        dummy_img = np.full((60, 400, 3), 100, dtype=np.uint8)

        cov = verify_mask_coverage(zero_mask, lines)
        assert cov == (False, None, 0)
        assert cov.mask_bounds == (0, 0, 0, 0)

        mock_rec = MagicMock()
        res = recover_truncated_text_lines(dummy_img, zero_mask, ocr_res, recognize_fn=mock_rec)
        assert mock_rec.call_count == 0
        assert res == ocr_res

    def test_full_white_mask_resilience(self):
        """Full-white masks must not crash, and monotonicity preserves original text if retry fails."""
        full_white = np.full((60, 400), 255, dtype=np.uint8)
        lines = [OCRTextLine(text="baseline_text", box=[[50, 15], [200, 15], [200, 45], [50, 45]])]
        ocr_res = OCRFrameResult(text="baseline_text", confidence=0.92, lines=lines)
        dummy_img = np.full((60, 400, 3), 200, dtype=np.uint8)

        cov = verify_mask_coverage(full_white, lines)
        # Full white mask bounds are detected safely
        assert cov.has_overflow is True

        # When recognizer returns lower confidence or empty text, original result is safely retained
        mock_rec = MagicMock(return_value=OCRFrameResult(text="", confidence=0.0, lines=[]))
        res = recover_truncated_text_lines(dummy_img, full_white, ocr_res, recognize_fn=mock_rec)
        assert res.text == "baseline_text"
        assert res.confidence == 0.92

    def test_noise_specks_filtered(self):
        """Small noise specks (< 10 px) must be ignored and not trigger false overflow."""
        mask = np.zeros((60, 500), dtype=np.uint8)
        # Genuine text line at x=100..400
        mask[20:40, 100:400] = 255
        # Noise specks of area 4 at far left and far right
        mask[20:22, 10:12] = 255
        mask[20:22, 480:482] = 255

        lines = [OCRTextLine(text="centered_text", box=[[95, 18], [405, 18], [405, 42], [95, 42]])]
        cov = verify_mask_coverage(mask, lines)
        assert cov.has_overflow is False
        assert cov.overflow_dir is None

    def test_vertical_band_filtering_ignores_external_watermarks(self):
        """Watermarks / logos outside subtitle vertical band must not trigger false overflow."""
        mask = np.zeros((120, 600), dtype=np.uint8)
        # Subtitle text line at y=80..105, x=150..450
        mask[80:105, 150:450] = 255
        # Large watermark at top-left: y=5..25, x=20..80 (area=1200)
        mask[5:25, 20:80] = 255

        lines = [OCRTextLine(text="bottom_subtitle", box=[[145, 80], [455, 80], [455, 105], [145, 105]])]
        cov = verify_mask_coverage(mask, lines)
        # The watermark at y=5..25 must be filtered out by vertical band filtering
        assert cov.has_overflow is False
        assert cov.overflow_dir is None

    def test_downscaled_mask_coordinate_space_normalization(self):
        """Mask with different resolution from frame is normalized correctly without scaling distortion."""
        full_img = np.full((120, 600, 3), 50, dtype=np.uint8)
        downscaled_mask = np.zeros((60, 300), dtype=np.uint8)
        # Overflow on left in downscaled coordinates (box at x=100, mask at x=20)
        downscaled_mask[10:30, 20:250] = 255

        truncated_lines = [OCRTextLine(text="PARTIAL_TEXT", box=[[100, 20], [500, 20], [500, 60], [100, 60]])]
        orig_res = OCRFrameResult(text="PARTIAL_TEXT", confidence=0.8, lines=truncated_lines)

        calls = []

        def mock_rec(strip, **kw):
            calls.append(strip.shape)
            return OCRFrameResult(
                text="FULL_RECOVERED_TEXT",
                confidence=0.92,
                lines=[OCRTextLine(text="FULL_RECOVERED_TEXT", box=[[0, 0], [strip.shape[1], 0], [strip.shape[1], strip.shape[0]], [0, strip.shape[0]]])],
            )

        res = recover_truncated_text_lines(full_img, downscaled_mask, orig_res, recognize_fn=mock_rec)
        assert len(calls) == 1
        assert res.text == "FULL_RECOVERED_TEXT"


# ==============================================================================
# 9. Milestone 1 Iteration 2: RapidOCR Parameter Override Freshness Stress
# ==============================================================================

class TestRapidOCRParameterOverrideFreshness:
    """Stress-test that consecutive calls with differing parameters never retain stale values."""

    def test_consecutive_parameter_overrides_on_cached_engine(self):
        cache = OCREngineCache()

        class MockPostProcess:
            def __init__(self):
                self.box_thresh = 0.5
                self.thresh = 0.3
                self.unclip_ratio = 1.6

        class MockDetector:
            def __init__(self):
                self.postprocess_op = MockPostProcess()

        class MockCfgDet:
            def __init__(self):
                self.box_thresh = 0.5
                self.thresh = 0.3
                self.unclip_ratio = 1.6

        class MockCfgGlobal:
            def __init__(self):
                self.text_score = 0.5

        class MockCfg:
            def __init__(self):
                self.Det = MockCfgDet()
                self.Global = MockCfgGlobal()

        class MockEngine:
            def __init__(self):
                self.text_score = 0.5
                self.text_det = MockDetector()
                self.cfg = MockCfg()

            def __call__(self, img, **kwargs):
                return ([[ [[10, 10], [50, 10], [50, 30], [10, 30]], "dyn_text", 0.9 ]], [0.01])

        mock_engine = MockEngine()
        cache._rapidocr = mock_engine

        test_configs = [
            {"box_thresh": 0.60, "thresh": 0.40, "unclip_ratio": 2.40, "text_score": 0.70},
            {"box_thresh": 0.20, "thresh": 0.15, "unclip_ratio": 1.50, "text_score": 0.25},
            {"box_thresh": 0.35, "thresh": 0.28, "unclip_ratio": 1.85, "text_score": 0.45},
            {"box_thresh": 0.10, "thresh": 0.10, "unclip_ratio": 1.10, "text_score": 0.15},
            {"box_thresh": 0.80, "thresh": 0.50, "unclip_ratio": 2.80, "text_score": 0.90},
        ]

        dummy_img = np.full((40, 100, 3), 200, dtype=np.uint8)

        for cfg in test_configs:
            res = cache.recognize_rapidocr(
                dummy_img,
                box_thresh=cfg["box_thresh"],
                thresh=cfg["thresh"],
                unclip_ratio=cfg["unclip_ratio"],
                text_score=cfg["text_score"],
            )
            assert res.text == "dyn_text"

            # Postprocess attributes must match current config exactly
            post = mock_engine.text_det.postprocess_op
            assert post.box_thresh == pytest.approx(cfg["box_thresh"])
            assert post.thresh == pytest.approx(cfg["thresh"])
            assert post.unclip_ratio == pytest.approx(cfg["unclip_ratio"])
            assert mock_engine.text_score == pytest.approx(cfg["text_score"])

            # Cfg dicts must also match
            assert mock_engine.cfg.Det.box_thresh == pytest.approx(cfg["box_thresh"])
            assert mock_engine.cfg.Det.thresh == pytest.approx(cfg["thresh"])
            assert mock_engine.cfg.Det.unclip_ratio == pytest.approx(cfg["unclip_ratio"])
            assert mock_engine.cfg.Global.text_score == pytest.approx(cfg["text_score"])


# ==============================================================================
# 10. Milestone 1 Iteration 2: MaskCoverageResult Contract Stress
# ==============================================================================

class TestMaskCoverageResultContractStress:
    """Stress-test MaskCoverageResult unpacking, iteration, indexing, and equality."""

    def test_tuple_unpacking_and_equality_contract(self):
        res = MaskCoverageResult(True, "left", 35, (10, 20, 150, 60))

        # 3-element unpacking
        has_cov, direction, px = res
        assert (has_cov, direction, px) == (True, "left", 35)

        # 4-element unpacking
        has_cov, direction, px, bounds = res
        assert (has_cov, direction, px, bounds) == (True, "left", 35, (10, 20, 150, 60))

        # Equality with 3-tuple
        assert res == (True, "left", 35)

        # Equality with 4-tuple
        assert res == (True, "left", 35, (10, 20, 150, 60))

        # Named properties
        assert res.has_overflow is True
        assert res.overflow_dir == "left"
        assert res.overflow_px == 35
        assert res.mask_bounds == (10, 20, 150, 60)

        # Indexing 0..3
        assert res[0] is True
        assert res[1] == "left"
        assert res[2] == 35
        assert res[3] == (10, 20, 150, 60)

