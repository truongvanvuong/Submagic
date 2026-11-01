"""
Automated unit and integration tests for Phase 3:
- 3-Layer OpenCV Visual Filter (Luminance/Edge Mask, Stability Wait against Fade-in, Visual Diff skip)
- Multi-Engine OCR Adapters (RapidOCR, VietOCR, Gemini Vision API) with mocked models (no network hang)
- RapidFuzz Levenshtein Subtitle Deduplicator (merging adjacent noisy OCR frames & keeping sharpest crop)
- End-to-End `run_ocr_pipeline`, Crop Image saving, Exclusion Masking, SQLite auto-save, WebSocket events
- FastAPI `/api/jobs/ocr` and `/api/jobs/llm` endpoints
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import (
    ExtractionMode,
    LLMJobRequest,
    NormalizedBox,
    OCRJobRequest,
    ROIConfig,
    SubtitleSegment,
    WebSocketEvent,
)
from app.services.ffmpeg_service import crop_and_mask_roi, probe_video
from app.services.llm_service import run_llm_postprocess_pipeline
from app.services.ocr_engines import (
    OCREngineCache,
    OCRFrameResult,
    OCRTextLine,
    crop_text_box,
    order_and_merge_ocr_lines,
    resolve_ocr_engine,
)
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    ThreeLayerVisualFilter,
    compute_mask_visual_diff,
    compute_ocr_text_similarity,
    deduplicate_ocr_segments,
    extract_text_binary_mask,
    has_text_activity,
    run_ocr_pipeline,
)


def _draw_hardsub_roi(
    text: str,
    width: int = 480,
    height: int = 90,
    opacity: float = 1.0,
    bg_shift: int = 0,
) -> np.ndarray:
    """Create a synthetic ROI BGR image with a moving dark-to-mid background and bright hardsub text."""
    # Moving background pattern (luminance 30..110, below luma_threshold=185)
    x_grid = np.linspace(0, 255, width, dtype=np.float32)
    bg_row = np.clip(70 + 40 * np.sin((x_grid + bg_shift * 17) * 0.08), 20, 130).astype(np.uint8)

    bg_gray = np.tile(bg_row, (height, 1))
    roi_bgr = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)

    if not text:
        return roi_bgr

    overlay = roi_bgr.copy()
    # Draw dark stroke outline + bright white text fill
    cv2.putText(
        overlay,
        text,
        (24, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (10, 10, 10),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        text,
        (24, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    alpha = max(0.0, min(1.0, float(opacity)))
    return cv2.addWeighted(overlay, alpha, roi_bgr, 1.0 - alpha, 0.0)


def create_hardsub_test_video(dest_path: Path) -> Path:
    """
    Create a 3.0-second 10 FPS MP4 video (30 frames, 640x360) with:
    - Persistent watermark logo in top-right of ROI (to test Exclusion Mask)
    - Subtitle 1 (frames 2..11, 0.2s..1.2s): "Xin chao Sub Magic" (with fade-in at frame 2)
    - Blank interval (frames 12..14, 1.2s..1.5s)
    - Subtitle 2 (frames 15..26, 1.5s..2.7s): "Hardsub OCR Pipeline" (with slight flicker at frame 20)
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 10.0
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))

    for idx in range(30):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        # Moving background stripe in video
        stripe_x = (idx * 20) % width
        cv2.rectangle(frame, (stripe_x, 0), (min(width, stripe_x + 60), height), (95, 80, 70), -1)

        # Persistent bright watermark inside the right side of the subtitle ROI (x=520..600, y=285..335)
        cv2.putText(
            frame,
            "LOGO",
            (520, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Subtitle 1: frames 2..11 (0.2s -> 1.2s)
        if 2 <= idx <= 11:
            color = (195, 195, 195) if idx == 2 else (255, 255, 255)
            cv2.putText(
                frame,
                "Xin chao Sub Magic",
                (90, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 0, 0),
                5,
            )
            cv2.putText(
                frame,
                "Xin chao Sub Magic",
                (90, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                color,
                2,
            )

        # Subtitle 2: frames 15..26 (1.5s -> 2.7s), with a 1-frame brief drop at frame 20 to test RapidFuzz merge
        if 15 <= idx <= 26 and idx != 20:
            cv2.putText(
                frame,
                "Hardsub OCR Pipeline",
                (85, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 0, 0),
                5,
            )
            cv2.putText(
                frame,
                "Hardsub OCR Pipeline",
                (85, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
            )

        writer.write(frame)

    writer.release()
    return dest_path


def test_opencv_layer1_text_binary_mask_and_background_suppression():
    """Verify Layer 1 isolates bright text strokes while suppressing moving backgrounds and full-screen flashes."""
    # Empty frame -> no text activity
    blank_roi = _draw_hardsub_roi("", bg_shift=0)
    blank_mask = extract_text_binary_mask(blank_roi, luma_threshold=185)
    assert has_text_activity(blank_mask) is False

    # Same text on two different moving backgrounds -> masks should be nearly identical
    roi_bg1 = _draw_hardsub_roi("Sub Magic Layer 1", bg_shift=1)
    roi_bg2 = _draw_hardsub_roi("Sub Magic Layer 1", bg_shift=15)
    mask1 = extract_text_binary_mask(roi_bg1, luma_threshold=185)
    mask2 = extract_text_binary_mask(roi_bg2, luma_threshold=185)

    assert has_text_activity(mask1) is True
    assert has_text_activity(mask2) is True
    diff_same_text = compute_mask_visual_diff(mask1, mask2)
    assert diff_same_text < 0.08

    # Different text -> high visual diff
    roi_other = _draw_hardsub_roi("Mot cau phu de khac han", bg_shift=15)
    mask_other = extract_text_binary_mask(roi_other, luma_threshold=185)
    diff_other_text = compute_mask_visual_diff(mask1, mask_other)
    assert diff_other_text >= 0.25

    # Full-frame white flash should be rejected as non-subtitle
    white_flash = np.full((90, 480, 3), 255, dtype=np.uint8)
    flash_mask = extract_text_binary_mask(white_flash, luma_threshold=185)
    assert has_text_activity(flash_mask) is False


def test_opencv_layer2_and_layer3_filter_state_machine():
    """
    Verify:
    - Layer 2 waits `stability_frames` during fade-in, picks the sharpest 100% opacity frame,
      preserves the initial `start_time`, and ignores 1-frame transient flashes.
    - Layer 3 skips redundant OCR calls across identical text frames and extends `end_time`.
    """
    vfilter = ThreeLayerVisualFilter(
        luma_threshold=180,
        stability_frames=1,
        visual_diff_threshold=0.14,
        frame_step_sec=0.2,
    )

    completed_spans = []

    # t=0.0: Blank
    completed_spans.extend(vfilter.process_frame(_draw_hardsub_roi(""), 0.0))

    # t=0.2: Transient 1-frame flash that disappears at t=0.4 -> should be filtered out by Layer 2
    completed_spans.extend(vfilter.process_frame(_draw_hardsub_roi("Transient Flash"), 0.2))
    completed_spans.extend(vfilter.process_frame(_draw_hardsub_roi(""), 0.4))
    assert len(completed_spans) == 0
    assert vfilter.ocr_triggers_count == 0

    # t=0.6: Subtitle A starts fading in (opacity=0.65 -> partial mask)
    faded_frame = _draw_hardsub_roi("Hello World Subtitle", opacity=0.65, bg_shift=1)
    completed_spans.extend(vfilter.process_frame(faded_frame, 0.6))

    # t=0.8..1.6 (5 frames): Subtitle A at 100% opacity over moving background
    for step_idx, t in enumerate((0.8, 1.0, 1.2, 1.4, 1.6)):
        crisp_frame = _draw_hardsub_roi("Hello World Subtitle", opacity=1.0, bg_shift=step_idx + 2)
        completed_spans.extend(vfilter.process_frame(crisp_frame, t))

    # Direct cut at t=1.8..2.2 (3 frames) to Subtitle B without blank frame
    for step_idx, t in enumerate((1.8, 2.0, 2.2)):
        sub_b_frame = _draw_hardsub_roi("Second Distinct Line", opacity=1.0, bg_shift=step_idx)
        completed_spans.extend(vfilter.process_frame(sub_b_frame, t))

    # Flush at end of stream
    completed_spans.extend(vfilter.flush())

    assert len(completed_spans) == 2
    assert vfilter.ocr_triggers_count == 2
    assert vfilter.frames_skipped_by_layer3 >= 4

    span_a, span_b = completed_spans[0], completed_spans[1]
    # Initial start_time at t=0.6 preserved even though fade-in settled at t=0.8..1.0!
    assert span_a.start_time == 0.6
    assert span_a.end_time == 1.8
    # Best frame must be a 100% opacity frame (max pixel value >= 245), not the 0.65 faded frame
    assert int(span_a.best_frame_bgr.max()) >= 245

    assert span_b.start_time == 1.8
    assert span_b.end_time == 2.4


def test_ocr_engines_rapidocr_vietocr_gemini_with_mocks(monkeypatch):
    """Test RapidOCR, VietOCR, and Gemini Vision adapters using mocks so no model weights are downloaded."""
    assert resolve_ocr_engine("auto", "vi") == "vietocr"
    assert resolve_ocr_engine("auto", "ch") == "rapidocr"
    assert resolve_ocr_engine("auto", "en") == "rapidocr"
    assert resolve_ocr_engine("gemini", "vi") == "gemini"

    cache = OCREngineCache()
    sample_img = _draw_hardsub_roi("Xin chao Viet Nam")

    # 1. Mock RapidOCR returning two text boxes (two lines) + 1 low-confidence noise box
    def fake_rapidocr_call(img):
        return (
            [
                (
                    [[20, 45], [220, 45], [220, 75], [20, 75]],
                    "Dong thu hai",
                    0.92,
                ),
                (
                    [[20, 10], [200, 10], [200, 38], [20, 38]],
                    "Dong thu nhat",
                    0.96,
                ),
                (
                    [[400, 5], [420, 5], [420, 15], [400, 15]],
                    "noise",
                    0.15,  # Below min_confidence=0.35 -> filtered out
                ),
            ],
            [0.01, 0.01, 0.01],
        )

    monkeypatch.setattr(cache, "get_rapidocr", lambda: fake_rapidocr_call)

    rapid_res = cache.recognize_frame(sample_img, engine="rapidocr", language="en")
    assert rapid_res.engine_used == "rapidocr"
    # Sorted top-to-bottom (y=10 before y=45) and noise filtered out
    assert rapid_res.text == "Dong thu nhat\nDong thu hai"
    assert len(rapid_res.lines) == 2
    assert rapid_res.confidence == pytest.approx(0.94, abs=1e-2)

    # 2. Mock VietOCR Predictor (used with RapidOCR box detector)
    class FakeVietOCRPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, pil_img, return_prob=True):
            self.calls += 1
            if self.calls == 1:
                return ("Dòng thứ nhất", 0.98)
            return ("Dòng thứ hai", 0.95)

    fake_viet = FakeVietOCRPredictor()
    monkeypatch.setattr(cache, "get_vietocr", lambda: fake_viet)

    viet_res = cache.recognize_frame(sample_img, engine="auto", language="vi")
    assert viet_res.engine_used == "vietocr"
    assert viet_res.text == "Dòng thứ nhất\nDòng thứ hai"
    assert viet_res.confidence == pytest.approx(0.965, abs=1e-3)
    assert fake_viet.calls == 2

    # 3. Test VietOCR graceful fallback when VietOCR is not installed
    def raise_import_error():
        raise ImportError("No module named 'vietocr'")

    monkeypatch.setattr(cache, "get_vietocr", raise_import_error)
    fallback_res = cache.recognize_frame(sample_img, engine="vietocr", language="vi")
    assert fallback_res.engine_used == "rapidocr"
    assert fallback_res.text == "Dong thu nhat\nDong thu hai"

    # 4. Mock Gemini Vision OCR batch & single frame
    monkeypatch.setattr(
        cache,
        "recognize_gemini_batch",
        lambda imgs, language="vi", api_key=None: [
            OCRFrameResult(
                text="Phụ đề từ Gemini Cloud",
                confidence=0.99,
                lines=[OCRTextLine(text="Phụ đề từ Gemini Cloud", confidence=0.99)],
                engine_used="gemini",
            )
            for _ in imgs
        ],
    )
    gem_res = cache.recognize_frame(
        sample_img, engine="gemini", language="vi", api_key="dummy-key"
    )
    assert gem_res.engine_used == "gemini"
    assert gem_res.text == "Phụ đề từ Gemini Cloud"
    assert gem_res.confidence == 0.99


def test_rapidfuzz_subtitle_deduplicator():
    """Verify RapidFuzz merges adjacent segments with >= 85% similarity and small gap, keeping highest confidence."""
    raw_segments = [
        SubtitleSegment(
            id="s1",
            project_id="p1",
            index=1,
            start_time=1.0,
            end_time=2.0,
            text="Xin chào các bạr",  # Slight OCR typo, lower confidence
            confidence=0.81,
            source="ocr",
            crop_image_url="/static/crops/p1/crop_blurry.jpg",
        ),
        SubtitleSegment(
            id="s2",
            project_id="p1",
            index=2,
            start_time=2.15,  # Gap = 0.15s <= 0.40s
            end_time=3.4,
            text="Xin chào các bạn",  # Cleaner OCR recognition, higher confidence
            confidence=0.97,
            source="ocr",
            crop_image_url="/static/crops/p1/crop_sharp.jpg",
        ),
        SubtitleSegment(
            id="s3",
            project_id="p1",
            index=3,
            start_time=3.5,
            end_time=5.0,
            text="Chào mừng đến với Sub Magic!",  # Different sentence -> not merged
            confidence=0.93,
            source="ocr",
            crop_image_url="/static/crops/p1/crop_3.jpg",
        ),
        SubtitleSegment(
            id="s4",
            project_id="p1",
            index=4,
            start_time=8.0,  # Gap = 3.0s > 0.40s -> not merged even if same text
            end_time=9.2,
            text="Chào mừng đến với Sub Magic!",
            confidence=0.95,
            source="ocr",
            crop_image_url="/static/crops/p1/crop_4.jpg",
        ),
    ]

    sim = compute_ocr_text_similarity("Xin chào các bạr", "Xin chào các bạn")
    assert sim >= 85.0

    merged_list = deduplicate_ocr_segments(
        raw_segments, fuzzy_match_threshold=85.0, max_merge_gap_sec=0.40
    )
    assert len(merged_list) == 3

    # First two merged: start=1.0, end=3.4, winning text="Xin chào các bạn", winning crop="crop_sharp.jpg"
    assert merged_list[0].index == 1
    assert merged_list[0].start_time == 1.0
    assert merged_list[0].end_time == 3.4
    assert merged_list[0].text == "Xin chào các bạn"
    assert merged_list[0].confidence == 0.97
    assert merged_list[0].crop_image_url == "/static/crops/p1/crop_sharp.jpg"

    assert merged_list[1].index == 2
    assert merged_list[1].start_time == 3.5
    assert merged_list[1].end_time == 5.0

    assert merged_list[2].index == 3
    assert merged_list[2].start_time == 8.0
    assert merged_list[2].end_time == 9.2


@pytest.mark.asyncio
async def test_end_to_end_ocr_pipeline_and_llm_postprocessor(tmp_path: Path, monkeypatch):
    """
    End-to-end test of `run_ocr_pipeline` on a synthetic hardsub video:
    - ROI cropping & Exclusion Masking over the watermark
    - 3-Layer OpenCV filtering
    - Mocked OCR engine returning noisy split segments that get merged via RapidFuzz
    - Saving crop images to disk
    - Real-time WebSocket events & SQLite persistence
    - Followed by Gemini LLM post-processor (`run_llm_postprocess_pipeline`) with mocked API call
    """
    from app.services import llm_service, ocr_pipeline

    await db.init_db()
    video_file = create_hardsub_test_video(tmp_path / "hardsub_e2e.mp4")
    meta = probe_video(video_file)

    roi_cfg = ROIConfig(
        main_roi=NormalizedBox(x=0.10, y=0.75, w=0.85, h=0.22),
        # Mask out the "LOGO" watermark at x=520..600 (0.81..0.94), y=285..335 (0.79..0.93)
        exclusion_masks=[NormalizedBox(x=0.80, y=0.76, w=0.16, h=0.20, label="Logo Mask")],
    )

    project = await db.create_project(
        name="OCR E2E Demo",
        video_path=str(video_file),
        video_metadata=meta,
        mode=ExtractionMode.OCR,
        language="vi",
        roi_config=roi_cfg,
    )

    # Mock `ocr_engine_cache.recognize_frame` to return deterministic results for the spans
    call_counter = {"count": 0}

    def fake_recognize_frame(image_bgr, engine="auto", language="vi", api_key=None, min_confidence=0.35):
        call_counter["count"] += 1
        idx = call_counter["count"]
        if idx == 1:
            return OCRFrameResult(
                text="Xin chào Sub Magic",
                confidence=0.95,
                lines=[OCRTextLine(text="Xin chào Sub Magic", confidence=0.95)],
                engine_used="vietocr",
            )
        if idx == 2:
            # First part of Subtitle 2 before the 1-frame flicker at frame 20
            return OCRFrameResult(
                text="Hardsub OCR Pipelinc",  # 1-char typo, lower confidence
                confidence=0.84,
                lines=[OCRTextLine(text="Hardsub OCR Pipelinc", confidence=0.84)],
                engine_used="vietocr",
            )
        # Second part of Subtitle 2 after the 1-frame flicker -> should merge with idx==2 via RapidFuzz!
        return OCRFrameResult(
            text="Hardsub OCR Pipeline",
            confidence=0.98,
            lines=[OCRTextLine(text="Hardsub OCR Pipeline", confidence=0.98)],
            engine_used="vietocr",
        )

    monkeypatch.setattr(ocr_pipeline.ocr_engine_cache, "recognize_frame", fake_recognize_frame)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent) -> None:
        events.append(ev)

    req = OCRJobRequest(
        project_id=project.id,
        language="vi",
        engine="auto",
        sample_fps=10.0,
        roi_config=roi_cfg,
        luma_threshold=180,
        stability_frames=1,
        visual_diff_threshold=0.14,
        fuzzy_match_threshold=85.0,
        max_merge_gap_sec=0.40,
        save_crop_images=True,
    )

    segments = await run_ocr_pipeline(req=req, job_id="ocr-job-1", emit=collect_event)

    # Even though 3 spans were OCR'd (due to the 1-frame flicker in Subtitle 2),
    # RapidFuzz merged spans 2 & 3 into a single segment!
    assert call_counter["count"] == 3
    assert len(segments) == 2
    assert segments[0].text == "Xin chào Sub Magic"
    assert segments[1].text == "Hardsub OCR Pipeline"
    assert segments[1].confidence == 0.98
    assert segments[0].crop_image_url is not None
    assert segments[0].crop_image_url.startswith(f"/static/crops/{project.id}/")

    event_types = [e.event for e in events]
    assert "job_started" in event_types
    assert "segment_created" in event_types
    assert "segment_updated" in event_types  # Triggered when span 3 merged into span 2
    assert "job_completed" in event_types

    # Verify SQLite persistence
    saved_proj = await db.get_project(project.id)
    assert saved_proj is not None
    assert saved_proj.status.value == "completed"
    assert len(saved_proj.segments) == 2
    assert saved_proj.segments[1].text == "Hardsub OCR Pipeline"

    # Now test LLM Post-Processor (spellcheck + bilingual translation) with mocked Gemini call
    monkeypatch.setattr(
        llm_service,
        "_call_gemini_batch_postprocess",
        lambda batch, fix_ocr_spelling, translate_to, api_key: [
            {
                "id": seg.id,
                "text": seg.text,
                "translated_text": f"[EN] {seg.text}",
            }
            for seg in batch
        ],
    )
    llm_events: list[WebSocketEvent] = []

    async def collect_llm_event(ev: WebSocketEvent) -> None:
        llm_events.append(ev)

    llm_req = LLMJobRequest(
        project_id=project.id,
        fix_ocr_spelling=True,
        translate_to="en",
        gemini_api_key="dummy-gemini-key",
    )
    updated_segs = await run_llm_postprocess_pipeline(
        req=llm_req, job_id="llm-job-1", emit=collect_llm_event
    )
    assert len(updated_segs) == 2
    assert updated_segs[0].translated_text == "[EN] Xin chào Sub Magic"
    assert updated_segs[1].translated_text == "[EN] Hardsub OCR Pipeline"


def test_jobs_api_ocr_endpoint(tmp_path: Path, monkeypatch):
    """Verify POST /api/jobs/ocr launches the background OCR pipeline via FastAPI."""
    from app.api import jobs as jobs_api

    video_file = create_hardsub_test_video(tmp_path / "api_ocr.mp4")

    async def dummy_run_ocr_pipeline(req, job_id, emit, cancel_event=None):
        return []

    monkeypatch.setattr(jobs_api, "run_ocr_pipeline", dummy_run_ocr_pipeline)

    with TestClient(app) as client:
        create_res = client.post(
            "/api/projects",
            json={
                "name": "API OCR Test",
                "video_path": str(video_file),
                "mode": "ocr",
                "language": "vi",
            },
        )
        assert create_res.status_code == 200
        proj_id = create_res.json()["id"]

        ocr_res = client.post(
            "/api/jobs/ocr",
            json={
                "project_id": proj_id,
                "language": "vi",
                "engine": "auto",
                "sample_fps": 6.0,
            },
        )
        assert ocr_res.status_code == 200
        body = ocr_res.json()
        assert body["project_id"] == proj_id
        assert body["status"] == "running"
        assert "job_id" in body

        # Missing project should return 404
        not_found_res = client.post(
            "/api/jobs/ocr",
            json={"project_id": "non-existent-id", "language": "vi"},
        )
        assert not_found_res.status_code == 404


@pytest.mark.asyncio
async def test_ocr_edge_cases_tilted_crop_same_row_cancellation_and_gemini_client(
    tmp_path: Path, monkeypatch
):
    """Test edge cases: empty inputs, tilted polygon perspective warp, same-row boxes, Gemini Client JSON parsing, and job cancellation."""
    import asyncio
    from google import genai

    cache = OCREngineCache()

    # 1. Empty / zero-size image handling
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert cache.recognize_rapidocr(empty_img).text == ""
    assert cache.recognize_vietocr(empty_img).text == ""
    assert cache.recognize_gemini_batch([]) == []

    # 2. Tilted 4-point box perspective warp vs horizontal box crop
    canvas = np.full((100, 300, 3), 128, dtype=np.uint8)
    tilted_box = [[20.0, 20.0], [180.0, 50.0], [175.0, 80.0], [15.0, 50.0]]
    warped_crop = crop_text_box(canvas, tilted_box)
    assert warped_crop.shape[0] >= 8 and warped_crop.shape[1] >= 8

    # 3. Same-row horizontal box merging (left-to-right) + multi-row sorting
    lines = [
        OCRTextLine(
            text="bạn!",
            confidence=0.95,
            box=[[150, 12], [220, 12], [220, 36], [150, 36]],
        ),
        OCRTextLine(
            text="Xin chào",
            confidence=0.96,
            box=[[20, 10], [130, 10], [130, 35], [20, 35]],
        ),
    ]
    merged_txt, avg_c, ordered = order_and_merge_ocr_lines(lines)
    assert merged_txt == "Xin chào bạn!"
    assert len(ordered) == 2

    # 4. Gemini batch OCR internal client call & missing API key error
    monkeypatch.setattr("app.services.ocr_engines.settings.gemini_api_key", None)
    with pytest.raises(ValueError):
        cache.recognize_gemini(_draw_hardsub_roi("Test"), language="vi", api_key=None)

    class DummyResponse:
        text = json.dumps(
            [{"index": 0, "text": "  Dòng Gemini 1  \n  Dòng 2 ", "confidence": 0.97}]
        )

    class DummyModels:
        def generate_content(self, model, contents, config):
            return DummyResponse()

    class DummyGenAIClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.models = DummyModels()

    monkeypatch.setattr(genai, "Client", DummyGenAIClient)
    gem_batch = cache.recognize_gemini_batch(
        [_draw_hardsub_roi("Test")], language="vi", api_key="test-key-123"
    )
    assert len(gem_batch) == 1
    assert gem_batch[0].text == "Dòng Gemini 1\nDòng 2"
    assert gem_batch[0].confidence == 0.97

    # 5. Cancellation during run_ocr_pipeline
    await db.init_db()
    video_file = create_hardsub_test_video(tmp_path / "cancel_ocr.mp4")
    proj = await db.create_project(
        name="Cancel OCR Test",
        video_path=str(video_file),
        mode=ExtractionMode.OCR,
        language="vi",
    )

    cancel_ev = asyncio.Event()
    cancel_ev.set()  # Pre-set cancellation
    events: list[WebSocketEvent] = []

    async def collect_ev(ev: WebSocketEvent) -> None:
        events.append(ev)

    req = OCRJobRequest(project_id=proj.id, language="vi", stability_frames=0)
    await run_ocr_pipeline(req=req, job_id="job-cancel", emit=collect_ev, cancel_event=cancel_ev)
    assert any(e.event == "job_cancelled" for e in events)
    cancelled_proj = await db.get_project(proj.id)
    assert cancelled_proj is not None
    assert cancelled_proj.status.value == "cancelled"


def test_stabilizing_fade_in_out_karaoke_flush_and_vietocr_fallback(monkeypatch):
    """
    Regression tests for edge cases in ThreeLayerVisualFilter and VietOCR fallback:
    1. Multi-frame fade-in -> crisp -> fade-out -> blank that ends while still in STABILIZING
       must emit the span and keep the sharpest 100%-opacity frame (not overwrite with fade-out).
    2. Karaoke / word-by-word accumulation changing mask continuously for > 0.65s must emit
       the accumulated span rather than overwriting start_time and losing the earlier subtitle.
    3. A 1-frame transient flash on the very last frame of the video must be rejected by flush().
    4. recognize_vietocr must fallback to full ROI when RapidOCR finds no sub-boxes on a tight single-line crop.
    """
    # 1. Fade-in (t=0.0, opacity=0.65) -> Crisp (t=0.2, opacity=1.0) -> Fade-out (t=0.4, opacity=0.60) -> Blank (t=0.6)
    # Note: because mask changes between 0.65 -> 1.0 -> 0.60, _stable_count stays 0 in STABILIZING,
    # but sampled_frames=3 > stability_frames=1, so it is a real subtitle and must be emitted at t=0.6 with the crisp frame!
    vf = ThreeLayerVisualFilter(
        luma_threshold=180,
        stability_frames=1,
        visual_diff_threshold=0.05,
        frame_step_sec=0.2,
    )
    spans = []
    spans.extend(vf.process_frame(_draw_hardsub_roi("Short Fading Line", opacity=0.65), 0.0))
    spans.extend(vf.process_frame(_draw_hardsub_roi("Short Fading Line", opacity=1.0), 0.2))
    spans.extend(vf.process_frame(_draw_hardsub_roi("Short Fading Line", opacity=0.60), 0.4))
    spans.extend(vf.process_frame(_draw_hardsub_roi(""), 0.6))

    assert len(spans) == 1
    assert spans[0].start_time == 0.0
    assert spans[0].end_time == 0.6
    # Best frame must be the crisp 1.0 opacity frame at t=0.2 (max pixel >= 245), not the 0.60 fade-out frame!
    assert int(spans[0].best_frame_bgr.max()) >= 245

    # 2. Continuous word-by-word build-up (karaoke style) exceeding 0.65s
    vf_karaoke = ThreeLayerVisualFilter(
        luma_threshold=180,
        stability_frames=1,
        visual_diff_threshold=0.08,
        frame_step_sec=0.2,
    )
    k_spans = []
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Xin"), 0.0))
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Xin chao"), 0.2))
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Xin chao cac"), 0.4))
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Xin chao cac ban"), 0.6))
    # At t=0.8 (elapsed 0.8s > 0.65s), the accumulated span [0.0, 0.8] must be emitted instead of lost!
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Cau tiep theo moi"), 0.8))
    k_spans.extend(vf_karaoke.process_frame(_draw_hardsub_roi("Cau tiep theo moi"), 1.0))
    k_spans.extend(vf_karaoke.flush())

    assert len(k_spans) == 2
    assert k_spans[0].start_time == 0.0
    assert k_spans[0].end_time == 0.8
    assert k_spans[1].start_time == 0.8
    assert k_spans[1].end_time == 1.2

    # 3. 1-frame transient flash on the very last frame before flush() must be rejected when stability_frames=1
    vf_end = ThreeLayerVisualFilter(
        luma_threshold=180,
        stability_frames=1,
        visual_diff_threshold=0.14,
        frame_step_sec=0.2,
    )
    assert vf_end.process_frame(_draw_hardsub_roi("Last Frame Noise Flash"), 2.8) == []
    assert vf_end.flush() == []
    assert vf_end.ocr_triggers_count == 0

    # 4. VietOCR fallback to full ROI when RapidOCR detector returns no sub-boxes on a tight single-line image
    cache = OCREngineCache()
    monkeypatch.setattr(cache, "get_rapidocr", lambda: (lambda img: ([], [])))

    class SingleLineVietPredictor:
        def predict(self, pil_img, return_prob=True):
            return ("Phụ đề dòng đơn sát viền", 0.96)

    monkeypatch.setattr(cache, "get_vietocr", lambda: SingleLineVietPredictor())
    tight_res = cache.recognize_vietocr(_draw_hardsub_roi("Phu de dong don"), min_confidence=0.35)
    assert tight_res.engine_used == "vietocr"
    assert tight_res.text == "Phụ đề dòng đơn sát viền"
    assert tight_res.confidence == 0.96

