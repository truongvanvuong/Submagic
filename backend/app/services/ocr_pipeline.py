"""
Phase 3: Hardsub OCR Pipeline & 3-Layer OpenCV Visual Filter (ocr_pipeline.py).
Features:
- Adaptive Frame Sampler (5-10 FPS) with ROI cropping and Exclusion Mask black-out
- OpenCV 3-Layer Visual Filter:
  * Layer 1: High-contrast stroke/edge & luminance text binary mask (removes moving background)
  * Layer 2: Stability check (waits `stability_frames` to avoid OCRing fade-in/pop-up frames while preserving initial `start_time`)
  * Layer 3: Visual diff (Dice/IoU mask comparison) skipping redundant OCR calls and extending `end_time`
- Hybrid OCR execution (RapidOCR, VietOCR, Gemini Vision API)
- RapidFuzz Levenshtein deduplication (>= 85% similarity & gap <= max_merge_gap_sec) keeping highest-confidence text & sharpest crop image
- Real-time SQLite auto-save and WebSocket streaming
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np
from rapidfuzz import fuzz

from app.config import settings
from app.database import db
from app.schemas import (
    ExportFormat,
    ExtractionMode,
    JobStatus,
    OCRJobRequest,
    SubtitleSegment,
    SubtitleSegmentUpdate,
    WebSocketEvent,
)
from app.services.ffmpeg_service import crop_and_mask_roi, probe_video, resolve_media_path
from app.services.ocr_engines import (
    clean_ocr_text,
    detect_text_language,
    ocr_engine_cache,
    resolve_ocr_engine,
)
from app.utils.subtitle_formatter import format_subtitles

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]


def extract_text_binary_mask(
    roi_bgr: np.ndarray,
    luma_threshold: int = 185,
) -> np.ndarray:
    """
    Layer 1 of 3-Layer OpenCV Filter:
    Extract a clean binary mask (0 or 255) of high-contrast subtitle strokes while
    rejecting ~90% of moving background scenery.
    Combines:
    1. Luminance threshold & morphological Top-Hat for bright text strokes.
    2. Canny/gradient stroke-edge adjacency so large uniform bright background regions
       (sky, walls, flashes) without internal character edges are suppressed.
    3. Vectorized Connected-component filtering (NumPy LUT) to remove tiny 1-2px noise specks.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if roi_bgr.ndim == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()

    orig_h, orig_w = gray.shape[:2]
    if orig_h < 4 or orig_w < 4:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Downscale for mask calculation if height > 80 (preserves contours while 4-5x faster)
    if orig_h > 80:
        scale = 72.0 / float(orig_h)
        work_w = max(16, int(round(orig_w * scale)))
        work_gray = cv2.resize(gray, (work_w, 72), interpolation=cv2.INTER_AREA)
    else:
        work_gray = gray

    h, w = work_gray.shape[:2]
    clamped_thresh = max(50, min(254, int(luma_threshold)))

    # 1. Direct luminance threshold for bright subtitle characters
    _, bright_mask = cv2.threshold(work_gray, clamped_thresh, 255, cv2.THRESH_BINARY)

    # 2. Top-Hat transform to isolate bright thin strokes over moderately bright backgrounds
    kw = max(9, min(31, (w // 20) | 1))
    kh = max(7, min(21, (h // 4) | 1))
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    tophat = cv2.morphologyEx(work_gray, cv2.MORPH_TOPHAT, tophat_kernel)
    tophat_cutoff = max(30, min(90, (255 - clamped_thresh) // 2 + 15))
    _, tophat_mask = cv2.threshold(tophat, tophat_cutoff, 255, cv2.THRESH_BINARY)

    # 3. Canny edge detection around character strokes
    edges = cv2.Canny(work_gray, 50, 150)
    edge_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_zone = cv2.dilate(edges, edge_dilate_kernel, iterations=1)

    # Bright pixels must sit near character edges (eliminates interior of large bright background patches)
    stroke_candidates = cv2.bitwise_and(bright_mask, edge_zone)
    tophat_bright = cv2.bitwise_and(bright_mask, tophat_mask)
    relaxed_thresh = max(135, clamped_thresh - 30)
    _, relaxed_bright = cv2.threshold(work_gray, relaxed_thresh, 255, cv2.THRESH_BINARY)
    tophat_edges = cv2.bitwise_and(
        cv2.bitwise_and(relaxed_bright, tophat_mask), edge_zone
    )
    combined = cv2.bitwise_or(
        stroke_candidates, cv2.bitwise_or(tophat_bright, tophat_edges)
    )

    # Close tiny 1-2px gaps inside character strokes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 4. Connected components filter (Vectorized NumPy LUT): remove isolated specks (< 6 px) and full-frame flash blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    comp_ws = stats[1:, cv2.CC_STAT_WIDTH]
    comp_hs = stats[1:, cv2.CC_STAT_HEIGHT]
    max_blob_area = int(h * w * 0.55)

    valid_mask = (
        (areas >= 6)
        & (areas <= max_blob_area)
        & ~((comp_ws >= int(w * 0.92)) & (comp_hs >= int(h * 0.90)))
    )
    valid_labels = np.where(valid_mask)[0] + 1
    if len(valid_labels) == 0:
        clean_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_labels] = 255
        clean_mask = lut[labels]

    if orig_h > 80:
        return cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return clean_mask


def has_text_activity(
    mask: np.ndarray,
    min_pixels: int = 12,
    min_ratio: float = 0.001,
    max_ratio: float = 0.65,
) -> bool:
    """Return True if the binary mask contains plausible subtitle text strokes."""
    if mask is None or mask.size == 0:
        return False
    nz = int(np.count_nonzero(mask))
    if nz < min_pixels:
        return False
    ratio = nz / float(mask.size)
    return min_ratio <= ratio <= max_ratio


def compute_mask_visual_diff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Layer 3 of 3-Layer OpenCV Filter:
    Compute visual dissimilarity in [0.0, 1.0] between two binary text masks.
    Uses 1-pixel dilated Dice/IoU similarity so minor compression jitter or background
    anti-aliasing around character borders produces diff near 0.0, whereas actual
    subtitle text changes produce diff >= `visual_diff_threshold`.
    """
    active_a = has_text_activity(mask_a)
    active_b = has_text_activity(mask_b)

    if not active_a and not active_b:
        return 0.0
    if active_a != active_b:
        return 1.0

    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(
            mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    bin_a = mask_a > 0
    bin_b = mask_b > 0

    total_fg = int(bin_a.sum()) + int(bin_b.sum())
    if total_fg == 0:
        return 0.0

    # 1px dilation tolerance for video compression jitter along text edges
    k3 = np.ones((3, 3), dtype=np.uint8)
    dil_a = cv2.dilate(mask_a, k3, iterations=1) > 0
    dil_b = cv2.dilate(mask_b, k3, iterations=1) > 0

    matched_a = int(np.logical_and(bin_a, dil_b).sum())
    matched_b = int(np.logical_and(bin_b, dil_a).sum())
    tolerant_dice = (matched_a + matched_b) / float(total_fg)

    # Also check bounding-width / pixel-count ratio so adding/removing a short word is caught
    count_a = float(bin_a.sum())
    count_b = float(bin_b.sum())
    size_ratio = min(count_a, count_b) / max(count_a, count_b)

    similarity = tolerant_dice * (0.75 + 0.25 * size_ratio)
    diff = 1.0 - max(0.0, min(1.0, similarity))
    return round(diff, 4)


def compute_frame_sharpness(
    roi_bgr: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """
    Estimate subtitle clarity/sharpness so full-opacity crisp frames are chosen over
    semi-transparent fade-in or motion-blurred frames.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    gray = (
        cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if roi_bgr.ndim == 3
        else roi_bgr
    )
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mask is not None and mask.shape == gray.shape and np.count_nonzero(mask) > 0:
        fg_luma = float(gray[mask > 0].mean())
        fg_pixels = float(np.count_nonzero(mask))
        return lap_var + (fg_luma * 2.5) + min(200.0, fg_pixels * 0.05)
    return lap_var


@dataclass
class VisualSubtitleSpan:
    """Represents a stabilized visual subtitle appearance interval ready for AI OCR."""
    start_time: float
    end_time: float
    best_frame_bgr: np.ndarray
    best_mask: np.ndarray
    best_sharpness: float = 0.0
    sampled_frames: int = 1


class ThreeLayerVisualFilter:
    """
    Stateful 3-Layer OpenCV Filter for Hardsub OCR:
    - Layer 1: High-Contrast Edge/Stroke & Luminance Masking (`extract_text_binary_mask`)
    - Layer 2: Stability Wait (`stability_frames`) to avoid OCR on fade-in/pop-up animations
      while preserving the exact initial `start_time` when the subtitle first appeared.
    - Layer 3: Visual Diff (`compute_mask_visual_diff`) to skip OCR when consecutive frames
      display the same text mask, simply extending `end_time`.
    """

    def __init__(
        self,
        luma_threshold: int = 185,
        stability_frames: int = 1,
        visual_diff_threshold: float = 0.14,
        frame_step_sec: float = 1.0 / 6.0,
    ) -> None:
        self.luma_threshold = luma_threshold
        self.stability_frames = max(0, int(stability_frames))
        self.visual_diff_threshold = float(visual_diff_threshold)
        self.frame_step_sec = max(0.02, float(frame_step_sec))

        # State machine: "IDLE", "STABILIZING", "ACTIVE"
        self._state: str = "IDLE"
        self._pending_span: VisualSubtitleSpan | None = None
        self._pending_last_mask: np.ndarray | None = None
        self._stable_count: int = 0

        self._active_span: VisualSubtitleSpan | None = None

        # Telemetry counters
        self.total_frames_sampled: int = 0
        self.frames_with_text: int = 0
        self.ocr_triggers_count: int = 0
        self.frames_skipped_by_layer3: int = 0

    def process_frame(
        self,
        roi_bgr: np.ndarray,
        timestamp_sec: float,
        precomputed_mask: np.ndarray | None = None,
    ) -> list[VisualSubtitleSpan]:
        """
        Process one sampled ROI frame at `timestamp_sec`.
        Returns a list of completed `VisualSubtitleSpan`s (0 or 1) that ended at this frame.
        """
        self.total_frames_sampled += 1
        t = round(max(0.0, float(timestamp_sec)), 3)
        next_t = round(t + self.frame_step_sec, 3)

        # Layer 1: Extract high-contrast text binary mask
        mask = (
            precomputed_mask
            if precomputed_mask is not None
            else extract_text_binary_mask(roi_bgr, luma_threshold=self.luma_threshold)
        )
        has_text = has_text_activity(mask)
        if has_text:
            self.frames_with_text += 1

        completed: list[VisualSubtitleSpan] = []

        if not has_text:
            # No text in current frame -> close active span or multi-frame stabilizing span, discard 1-frame flash
            if self._state == "ACTIVE" and self._active_span is not None:
                self._active_span.end_time = max(
                    round(self._active_span.start_time + 0.10, 3),
                    min(self._active_span.end_time, t),
                )
                completed.append(self._active_span)
            elif (
                self._state == "STABILIZING"
                and self._pending_span is not None
                and self._pending_span.sampled_frames > self.stability_frames
            ):
                self._pending_span.end_time = max(
                    round(self._pending_span.start_time + 0.10, 3),
                    min(self._pending_span.end_time, t),
                )
                self.ocr_triggers_count += 1
                completed.append(self._pending_span)
            self._reset_to_idle()
            return completed

        sharpness = compute_frame_sharpness(roi_bgr, mask)

        # State 1: IDLE -> text just appeared
        if self._state == "IDLE":
            self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)
            return completed

        # State 2: STABILIZING -> waiting for fade-in / pop-up animation to settle
        if self._state == "STABILIZING":
            assert self._pending_span is not None
            assert self._pending_last_mask is not None
            diff = compute_mask_visual_diff(self._pending_last_mask, mask)
            self._pending_span.end_time = next_t
            self._pending_span.sampled_frames += 1

            if diff < self.visual_diff_threshold:
                # Mask is consistent with previous frame
                self._stable_count += 1
                if sharpness >= self._pending_span.best_sharpness:
                    self._pending_span.best_frame_bgr = roi_bgr.copy()
                    self._pending_span.best_mask = mask
                    self._pending_span.best_sharpness = sharpness
                self._pending_last_mask = mask

                if self._stable_count >= self.stability_frames:
                    # Promoted to ACTIVE!
                    self._state = "ACTIVE"
                    self._active_span = self._pending_span
                    self._pending_span = None
                    self._pending_last_mask = None
                    self.ocr_triggers_count += 1
            else:
                # Still fading in or changing shape: if it has been changing for > 0.65s across
                # multiple frames (e.g. karaoke word build-up or fast cut), emit accumulated span first
                if (
                    (t - self._pending_span.start_time) > 0.65
                    and (self._pending_span.sampled_frames - 1) > self.stability_frames
                ):
                    self._pending_span.end_time = max(
                        round(self._pending_span.start_time + 0.15, 3), t
                    )
                    self.ocr_triggers_count += 1
                    completed.append(self._pending_span)
                    self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)
                    return completed

                if sharpness >= self._pending_span.best_sharpness:
                    self._pending_span.best_frame_bgr = roi_bgr.copy()
                    self._pending_span.best_mask = mask
                    self._pending_span.best_sharpness = sharpness
                self._pending_last_mask = mask
                self._stable_count = 0

            return completed

        # State 3: ACTIVE -> compare with active span mask (Layer 3 Visual Diff)
        if self._state == "ACTIVE":
            assert self._active_span is not None
            diff = compute_mask_visual_diff(self._active_span.best_mask, mask)
            if diff < self.visual_diff_threshold:
                # Layer 3 Hit: Same subtitle text! Skip OCR and extend end_time
                self.frames_skipped_by_layer3 += 1
                self._active_span.end_time = next_t
                self._active_span.sampled_frames += 1
                if sharpness > self._active_span.best_sharpness:
                    self._active_span.best_frame_bgr = roi_bgr.copy()
                    self._active_span.best_mask = mask
                    self._active_span.best_sharpness = sharpness
            else:
                # Subtitle changed directly to a new line without a blank frame in between
                self._active_span.end_time = max(
                    round(self._active_span.start_time + 0.15, 3), t
                )
                completed.append(self._active_span)
                self._active_span = None
                self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)

        return completed

    def flush(self) -> list[VisualSubtitleSpan]:
        """Finalize any active or pending subtitle span at the end of the video stream."""
        completed: list[VisualSubtitleSpan] = []
        if self._state == "ACTIVE" and self._active_span is not None:
            completed.append(self._active_span)
        elif (
            self._state == "STABILIZING"
            and self._pending_span is not None
            and self._pending_span.sampled_frames > self.stability_frames
        ):
            self.ocr_triggers_count += 1
            completed.append(self._pending_span)
        self._reset_to_idle()
        return completed

    def _start_new_candidate(
        self,
        roi_bgr: np.ndarray,
        mask: np.ndarray,
        sharpness: float,
        start_t: float,
        end_t: float,
    ) -> None:
        span = VisualSubtitleSpan(
            start_time=start_t,
            end_time=end_t,
            best_frame_bgr=roi_bgr.copy(),
            best_mask=mask,
            best_sharpness=sharpness,
            sampled_frames=1,
        )
        if self.stability_frames == 0:
            self._state = "ACTIVE"
            self._active_span = span
            self.ocr_triggers_count += 1
        else:
            self._state = "STABILIZING"
            self._pending_span = span
            self._pending_last_mask = mask
            self._stable_count = 0

    def _reset_to_idle(self) -> None:
        self._state = "IDLE"
        self._pending_span = None
        self._pending_last_mask = None
        self._stable_count = 0
        self._active_span = None


def normalize_for_fuzzy(text: str) -> str:
    """Normalize OCR text for Levenshtein comparison (lowercase, collapsed whitespace)."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


def compute_ocr_text_similarity(text_a: str, text_b: str) -> float:
    """Compute RapidFuzz normalized Levenshtein ratio (0.0 - 100.0) between two subtitle strings."""
    norm_a = normalize_for_fuzzy(text_a)
    norm_b = normalize_for_fuzzy(text_b)
    if not norm_a and not norm_b:
        return 100.0
    if not norm_a or not norm_b:
        return 0.0
    return float(fuzz.ratio(norm_a, norm_b))


class FuzzySubtitleDeduplicator:
    """
    Merges consecutive OCR subtitle segments whose normalized Levenshtein similarity
    (`rapidfuzz.fuzz.ratio`) is >= `fuzzy_match_threshold` (default 85%) and whose
    time gap is <= `max_merge_gap_sec` (default 0.20s).
    Automatically retains the text and `crop_image_url` of the higher-confidence recognition.
    """

    def __init__(
        self,
        fuzzy_match_threshold: float = 85.0,
        max_merge_gap_sec: float = 0.20,
    ) -> None:
        self.fuzzy_match_threshold = float(fuzzy_match_threshold)
        self.max_merge_gap_sec = float(max_merge_gap_sec)
        self.segments: list[SubtitleSegment] = []

    def add_or_merge(
        self, candidate: SubtitleSegment
    ) -> tuple[SubtitleSegment, bool]:
        """
        Add `candidate` segment or merge it into the previous segment if similar & adjacent.
        Returns `(segment, was_merged)`.
        """
        if not self.segments:
            candidate.index = 1
            self.segments.append(candidate)
            return candidate, False

        prev = self.segments[-1]
        gap = max(0.0, candidate.start_time - prev.end_time)
        similarity = compute_ocr_text_similarity(prev.text, candidate.text)

        if similarity >= self.fuzzy_match_threshold and gap <= self.max_merge_gap_sec:
            prev.start_time = round(min(prev.start_time, candidate.start_time), 3)
            prev.end_time = round(max(prev.end_time, candidate.end_time), 3)

            # Pick the text & crop image with higher confidence (or longer text on tie)
            cand_better = (candidate.confidence > prev.confidence) or (
                abs(candidate.confidence - prev.confidence) < 1e-4
                and len(candidate.text) > len(prev.text)
            )
            if cand_better:
                prev.text = candidate.text
                prev.confidence = round(candidate.confidence, 4)
                if candidate.crop_image_url:
                    prev.crop_image_url = candidate.crop_image_url

            return prev, True

        candidate.index = len(self.segments) + 1
        self.segments.append(candidate)
        return candidate, False


def deduplicate_ocr_segments(
    segments: list[SubtitleSegment],
    fuzzy_match_threshold: float = 85.0,
    max_merge_gap_sec: float = 0.20,
) -> list[SubtitleSegment]:
    """
    Standalone helper to deduplicate and merge a list of OCR `SubtitleSegment` items
    using RapidFuzz Levenshtein similarity and time-gap constraints.
    """
    dedup = FuzzySubtitleDeduplicator(
        fuzzy_match_threshold=fuzzy_match_threshold,
        max_merge_gap_sec=max_merge_gap_sec,
    )
    for seg in sorted(segments, key=lambda s: (s.start_time, s.index)):
        if not seg.text or not seg.text.strip():
            continue
        dedup.add_or_merge(seg.model_copy(deep=True))
    return dedup.segments


def save_roi_crop_image(
    roi_bgr: np.ndarray,
    project_id: str,
    segment_index: int,
    start_time: float,
    quality: int = 90,
) -> str | None:
    """
    Save the cropped ROI BGR frame to `data/crops/<project_id>/` and return its
    static URL (`/static/crops/<project_id>/<filename>`) for the interactive subtitle editor.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    proj_crop_dir = settings.crops_dir / project_id
    proj_crop_dir.mkdir(parents=True, exist_ok=True)

    ms_tag = int(round(max(0.0, start_time) * 1000))
    filename = f"ocr_{segment_index:04d}_{ms_tag}ms.jpg"
    out_path = proj_crop_dir / filename

    ok, buf = cv2.imencode(".jpg", roi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    out_path.write_bytes(buf.tobytes())
    return f"/static/crops/{project_id}/{filename}"


async def run_ocr_pipeline(
    req: OCRJobRequest,
    job_id: str,
    emit: EventCallback,
    cancel_event: asyncio.Event | None = None,
) -> list[SubtitleSegment]:
    """
    Execute the complete Hardsub OCR extraction pipeline for a project:
    1. Validate project, update ROI config, and inspect video metadata.
    2. Sample frames at `req.sample_fps`, crop ROI, and apply Exclusion Masks.
    3. Filter frames through the 3-Layer OpenCV Filter (Luminance/Edge Mask, Stability Wait, Visual Diff).
    4. Run OCR (RapidOCR / VietOCR / Gemini Vision API) only on distinct stabilized subtitle spans.
    5. Deduplicate & merge adjacent segments with RapidFuzz, saving crop images,
       auto-saving to SQLite, and broadcasting real-time WebSocket events.
    """
    cancel_ev = cancel_event or asyncio.Event()
    start_wall = time.perf_counter()

    project = await db.get_project(req.project_id)
    if not project:
        raise ValueError(f"Project not found: {req.project_id}")

    video_path = resolve_media_path(project.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_engine = resolve_ocr_engine(req.engine, req.language)

    await db.update_project_status(
        req.project_id,
        status=JobStatus.RUNNING,
        progress=1.0,
        mode=ExtractionMode.OCR,
        roi_config=req.roi_config,
    )
    await db.clear_project_segments(req.project_id)
    if req.save_crop_images:
        proj_crop_dir = settings.crops_dir / req.project_id
        if proj_crop_dir.exists():
            for old_crop in proj_crop_dir.glob("ocr_*.jpg"):
                try:
                    old_crop.unlink()
                except OSError:
                    pass

    await emit(
        WebSocketEvent(
            event="job_started",
            project_id=req.project_id,
            job_id=job_id,
            progress=1.0,
            message=f"Bắt đầu quét Hardsub OCR ({resolved_engine.upper()}) cho {video_path.name}...",
            extra={"engine": resolved_engine, "language": req.language},
        )
    )

    try:
        meta = project.video_metadata or await asyncio.to_thread(probe_video, video_path)
        video_fps = max(1.0, float(meta.fps or 25.0))
        duration_sec = max(0.5, float(meta.duration_sec or 1.0))
        sample_fps = max(1.0, min(video_fps, float(req.sample_fps)))
        frame_step = max(1, int(round(video_fps / sample_fps)))
        effective_step_sec = frame_step / video_fps

        visual_filter = ThreeLayerVisualFilter(
            luma_threshold=req.luma_threshold,
            stability_frames=req.stability_frames,
            visual_diff_threshold=req.visual_diff_threshold,
            frame_step_sec=effective_step_sec,
        )
        deduplicator = FuzzySubtitleDeduplicator(
            fuzzy_match_threshold=req.fuzzy_match_threshold,
            max_merge_gap_sec=req.max_merge_gap_sec,
        )

        loop = asyncio.get_running_loop()
        span_queue: asyncio.Queue[tuple[VisualSubtitleSpan, float] | dict | None] = (
            asyncio.Queue()
        )

        def _frame_scanner_worker() -> None:
            """Decode video frames, crop ROI + Exclusion Masks, and run 3-Layer OpenCV Filter with Coarse-to-Fine refinement."""
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                loop.call_soon_threadsafe(
                    span_queue.put_nowait,
                    {
                        "type": "error",
                        "error": RuntimeError(f"Không thể mở video: {video_path}"),
                    },
                )
                loop.call_soon_threadsafe(span_queue.put_nowait, None)
                return

            try:
                video_w = int(meta.width or 1280)
                video_h = int(meta.height or 720)
                rx1, ry1, rx2, ry2 = req.roi_config.main_roi.to_pixel_rect(video_w, video_h)
                ex_rects: list[tuple[int, int, int, int]] = []
                for em in req.roi_config.exclusion_masks:
                    mx1, my1, mx2, my2 = em.to_pixel_rect(video_w, video_h)
                    ix1 = max(0, mx1 - rx1)
                    iy1 = max(0, my1 - ry1)
                    ix2 = min(rx2 - rx1, mx2 - rx1)
                    iy2 = min(ry2 - ry1, my2 - ry1)
                    if ix2 > ix1 and iy2 > iy1:
                        ex_rects.append((ix1, iy1, ix2, iy2))

                def _fast_crop(frame: np.ndarray) -> np.ndarray:
                    fh, fw = frame.shape[:2]
                    if fw == video_w and fh == video_h:
                        cropped = frame[ry1:ry2, rx1:rx2]
                        if ex_rects:
                            cropped = cropped.copy()
                            for e_x1, e_y1, e_x2, e_y2 in ex_rects:
                                cropped[e_y1:e_y2, e_x1:e_x2] = 0
                        return cropped
                    return crop_and_mask_roi(frame, req.roi_config)

                frame_idx = 0
                last_progress_report = 0.0
                prev_coarse_idx: int | None = None
                prev_had_text: bool = False
                prev_coarse_mask: np.ndarray | None = None

                while not cancel_ev.is_set():
                    if frame_step > 1 and frame_idx % frame_step != 0:
                        if not cap.grab():
                            break
                        frame_idx += 1
                        continue

                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break

                    timestamp_sec = frame_idx / video_fps
                    roi_bgr = _fast_crop(frame_bgr)
                    curr_mask = extract_text_binary_mask(roi_bgr, luma_threshold=req.luma_threshold)
                    curr_has_text = has_text_activity(curr_mask)

                    refined_start_time: float | None = None
                    refined_end_time: float | None = None

                    # Coarse-to-fine boundary refinement on state transitions when frame_step > 1
                    if frame_step > 1 and prev_coarse_idx is not None:
                        start_win = prev_coarse_idx + 1
                        end_win = frame_idx - 1

                        if end_win >= start_win:
                            is_cut = (
                                prev_had_text
                                and curr_has_text
                                and prev_coarse_mask is not None
                                and compute_mask_visual_diff(prev_coarse_mask, curr_mask)
                                >= req.visual_diff_threshold
                            )
                            need_refine = (
                                (not prev_had_text and curr_has_text)
                                or (prev_had_text and not curr_has_text)
                                or is_cut
                            )

                            if need_refine:
                                try:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_win)
                                    window_rois: list[np.ndarray] = []
                                    for _ in range(start_win, frame_idx):
                                        ok_p, f_p = cap.read()
                                        if not ok_p or f_p is None:
                                            break
                                        window_rois.append(_fast_crop(f_p))
                                    cap.grab()  # Advance past frame_idx (already processed)

                                    if window_rois:
                                        # 1. Text appearance: blank -> text (binary search for first frame with text)
                                        if not prev_had_text and curr_has_text:
                                            low, high = 0, len(window_rois) - 1
                                            ans_offset = len(window_rois)
                                            while low <= high:
                                                mid = (low + high) // 2
                                                m_mid = extract_text_binary_mask(
                                                    window_rois[mid],
                                                    luma_threshold=req.luma_threshold,
                                                )
                                                if has_text_activity(m_mid):
                                                    ans_offset = mid
                                                    high = mid - 1
                                                else:
                                                    low = mid + 1
                                            found_idx = start_win + ans_offset
                                            refined_start_time = round(found_idx / video_fps, 3)

                                        # 2. Text disappearance: text -> blank (binary search for last frame with text)
                                        elif prev_had_text and not curr_has_text:
                                            low, high = 0, len(window_rois) - 1
                                            ans_offset = -1
                                            while low <= high:
                                                mid = (low + high) // 2
                                                m_mid = extract_text_binary_mask(
                                                    window_rois[mid],
                                                    luma_threshold=req.luma_threshold,
                                                )
                                                if has_text_activity(m_mid):
                                                    ans_offset = mid
                                                    low = mid + 1
                                                else:
                                                    high = mid - 1
                                            last_text_idx = (
                                                (start_win + ans_offset)
                                                if ans_offset >= 0
                                                else prev_coarse_idx
                                            )
                                            refined_end_time = round(
                                                (last_text_idx + 1) / video_fps, 3
                                            )

                                        # 3. Direct cut between subtitle lines (binary search for first frame differing from prev text)
                                        elif is_cut:
                                            assert prev_coarse_mask is not None
                                            low, high = 0, len(window_rois) - 1
                                            ans_offset = len(window_rois)
                                            while low <= high:
                                                mid = (low + high) // 2
                                                m_mid = extract_text_binary_mask(
                                                    window_rois[mid],
                                                    luma_threshold=req.luma_threshold,
                                                )
                                                if (
                                                    not has_text_activity(m_mid)
                                                    or compute_mask_visual_diff(
                                                        prev_coarse_mask, m_mid
                                                    )
                                                    >= req.visual_diff_threshold
                                                ):
                                                    ans_offset = mid
                                                    high = mid - 1
                                                else:
                                                    low = mid + 1
                                            cut_idx = start_win + ans_offset
                                            cut_t = round(cut_idx / video_fps, 3)
                                            refined_end_time = cut_t
                                            refined_start_time = cut_t
                                except Exception as e_refine:
                                    logger.debug("Refine error: %s", e_refine)
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)

                    completed_spans = visual_filter.process_frame(
                        roi_bgr, timestamp_sec, precomputed_mask=curr_mask
                    )

                    if refined_start_time is not None:
                        if visual_filter._state == "ACTIVE" and visual_filter._active_span is not None:
                            visual_filter._active_span.start_time = refined_start_time
                        elif visual_filter._state == "STABILIZING" and visual_filter._pending_span is not None:
                            visual_filter._pending_span.start_time = refined_start_time

                    if refined_end_time is not None:
                        for sp in completed_spans:
                            sp.end_time = max(round(sp.start_time + 0.05, 3), refined_end_time)

                    pct = min(95.0, 5.0 + (timestamp_sec / duration_sec) * 88.0)
                    for sp in completed_spans:
                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, pct))

                    prev_coarse_idx = frame_idx
                    prev_had_text = curr_has_text
                    prev_coarse_mask = curr_mask

                    if pct - last_progress_report >= 5.0 and not completed_spans:
                        last_progress_report = pct
                        elapsed_so_far = max(0.01, time.perf_counter() - start_wall)
                        rate = timestamp_sec / elapsed_so_far
                        eta_sec = max(0.0, (duration_sec - timestamp_sec) / max(0.01, rate))
                        loop.call_soon_threadsafe(
                            span_queue.put_nowait,
                            {
                                "type": "progress",
                                "progress": round(pct, 1),
                                "message": f"Đang quét khung hình OpenCV ({timestamp_sec:.1f}s/{duration_sec:.1f}s • Còn lại ~{int(round(eta_sec))}s)...",
                                "eta_sec": round(eta_sec, 1),
                            },
                        )

                    frame_idx += 1

                if not cancel_ev.is_set():
                    for sp in visual_filter.flush():
                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, 95.0))

                loop.call_soon_threadsafe(span_queue.put_nowait, None)
            except Exception as err:
                loop.call_soon_threadsafe(
                    span_queue.put_nowait, {"type": "error", "error": err}
                )
                loop.call_soon_threadsafe(span_queue.put_nowait, None)
            finally:
                cap.release()

        scanner_task = asyncio.create_task(asyncio.to_thread(_frame_scanner_worker))
        raw_crop_counter = 1

        try:
            stream_done = False
            while not stream_done:
                item = await span_queue.get()
                if item is None:
                    break

                if isinstance(item, dict):
                    if item.get("type") == "error":
                        raise item["error"]
                    if item.get("type") == "progress":
                        await emit(
                            WebSocketEvent(
                                event="job_progress",
                                project_id=req.project_id,
                                job_id=job_id,
                                progress=item["progress"],
                                message=item["message"],
                            )
                        )
                    continue

                if cancel_ev.is_set():
                    continue

                batch_spans: list[tuple[VisualSubtitleSpan, float]] = [item]
                # In Gemini Cloud mode, drain any additional ready spans from queue up to batch size 8
                if resolved_engine == "gemini":
                    while len(batch_spans) < 8 and not span_queue.empty():
                        queued_item = span_queue.get_nowait()
                        if queued_item is None:
                            stream_done = True
                            break
                        if isinstance(queued_item, dict):
                            if queued_item.get("type") == "error":
                                raise queued_item["error"]
                            continue
                        batch_spans.append(queued_item)

                if resolved_engine == "gemini" and len(batch_spans) > 1:
                    ocr_results = await asyncio.to_thread(
                        ocr_engine_cache.recognize_gemini_batch,
                        [sp.best_frame_bgr for sp, _ in batch_spans],
                        req.language,
                        req.gemini_api_key,
                    )
                else:
                    raw_bgr = batch_spans[0][0].best_frame_bgr
                    # Downscale only excessively high-resolution crops (e.g. 4K/2K) to preserve contour detail
                    if raw_bgr is not None and raw_bgr.shape[0] > 160:
                        scale = 120.0 / float(raw_bgr.shape[0])
                        new_w = max(16, int(round(raw_bgr.shape[1] * scale)))
                        target_bgr = cv2.resize(raw_bgr, (new_w, 120), interpolation=cv2.INTER_AREA)
                    else:
                        target_bgr = raw_bgr

                    single_res = await asyncio.to_thread(
                        ocr_engine_cache.recognize_frame,
                        target_bgr,
                        resolved_engine,
                        req.language,
                        req.gemini_api_key,
                    )
                    ocr_results = [single_res]

                for (span, pct), ocr_res in zip(batch_spans, ocr_results):
                    if cancel_ev.is_set():
                        break
                    cleaned_text = clean_ocr_text(ocr_res.text)
                    if not cleaned_text:
                        continue

                    crop_url: str | None = None
                    if req.save_crop_images:
                        crop_url = await asyncio.to_thread(
                            save_roi_crop_image,
                            span.best_frame_bgr,
                            req.project_id,
                            raw_crop_counter,
                            span.start_time,
                        )
                        raw_crop_counter += 1

                    candidate_seg = SubtitleSegment(
                        id=str(uuid.uuid4()),
                        project_id=req.project_id,
                        index=len(deduplicator.segments) + 1,
                        start_time=round(span.start_time, 3),
                        end_time=round(max(span.start_time + 0.05, span.end_time), 3),
                        text=cleaned_text,
                        confidence=round(float(ocr_res.confidence or 0.85), 4),
                        source="ocr",
                        crop_image_url=crop_url,
                        words=[],
                    )

                    active_seg, was_merged = deduplicator.add_or_merge(candidate_seg)
                    await db.update_project_status(
                        req.project_id, status=JobStatus.RUNNING, progress=pct
                    )

                    if was_merged and active_seg.id:
                        await db.update_segment(
                            req.project_id,
                            active_seg.id,
                            SubtitleSegmentUpdate(
                                start_time=active_seg.start_time,
                                end_time=active_seg.end_time,
                                text=active_seg.text,
                                confidence=active_seg.confidence,
                                crop_image_url=active_seg.crop_image_url,
                            ),
                        )
                        await emit(
                            WebSocketEvent(
                                event="segment_updated",
                                project_id=req.project_id,
                                job_id=job_id,
                                progress=round(pct, 1),
                                message=f"Gộp trùng lặp [{active_seg.start_time:.1f}s -> {active_seg.end_time:.1f}s] {active_seg.text.replace(chr(10), ' ')}",
                                segment=active_seg,
                            )
                        )
                    else:
                        await db.insert_segment(active_seg)
                        await emit(
                            WebSocketEvent(
                                event="segment_created",
                                project_id=req.project_id,
                                job_id=job_id,
                                progress=round(pct, 1),
                                message=f"[{active_seg.start_time:.1f}s -> {active_seg.end_time:.1f}s] {active_seg.text.replace(chr(10), ' ')}",
                                segment=active_seg,
                            )
                        )
        except Exception:
            cancel_ev.set()
            raise
        finally:
            await asyncio.gather(scanner_task, return_exceptions=True)

        if cancel_ev.is_set():
            await db.update_project_status(req.project_id, status=JobStatus.CANCELLED)
            await emit(
                WebSocketEvent(
                    event="job_cancelled",
                    project_id=req.project_id,
                    job_id=job_id,
                    message="Đã hủy tác vụ Hardsub OCR.",
                )
            )
            return deduplicator.segments

        elapsed = round(time.perf_counter() - start_wall, 2)
        await db.update_project_status(
            req.project_id, status=JobStatus.COMPLETED, progress=100.0
        )

        # Detect language across recognized segments
        all_text = " ".join(seg.text for seg in deduplicator.segments)
        det_lang = detect_text_language(all_text)
        if det_lang == "auto":
            det_lang = req.language if req.language != "auto" else "vi"

        # Auto-save export SRT file and record in extraction history
        export_srt_path = None
        try:
            srt_content = format_subtitles(deduplicator.segments, ExportFormat.SRT)
            safe_name = re.sub(r'[\\/*?:"<>|]', '_', project.name)
            srt_filename = f"{safe_name}_{req.project_id[:8]}.srt"
            srt_file = settings.exports_dir / srt_filename
            srt_file.write_text(srt_content, encoding="utf-8")
            export_srt_path = str(srt_file)
        except Exception as export_err:
            logger.warning("Could not auto-export history SRT: %s", export_err)

        try:
            await db.add_history_record(
                project_id=req.project_id,
                mode="ocr",
                language=req.language,
                detected_language=det_lang,
                segment_count=len(deduplicator.segments),
                export_srt_path=export_srt_path,
                duration_sec=duration_sec,
                elapsed_sec=elapsed,
            )
        except Exception as hist_err:
            logger.warning("Could not add extraction history record: %s", hist_err)

        await emit(
            WebSocketEvent(
                event="job_completed",
                project_id=req.project_id,
                job_id=job_id,
                progress=100.0,
                message=(
                    f"Hoàn tất Hardsub OCR: {len(deduplicator.segments)} câu sub trong {elapsed}s "
                    f"(Lọc bỏ {visual_filter.frames_skipped_by_layer3}/{visual_filter.total_frames_sampled} frame trùng)."
                ),
                extra={
                    "segment_count": len(deduplicator.segments),
                    "elapsed_sec": elapsed,
                    "total_frames_sampled": visual_filter.total_frames_sampled,
                    "frames_with_text": visual_filter.frames_with_text,
                    "ocr_triggers_count": visual_filter.ocr_triggers_count,
                    "frames_skipped_by_layer3": visual_filter.frames_skipped_by_layer3,
                    "engine": resolved_engine,
                    "detected_language": det_lang,
                },
            )
        )
        return deduplicator.segments

    except Exception as exc:
        logger.exception("OCR Pipeline failed for project %s", req.project_id)
        await db.update_project_status(req.project_id, status=JobStatus.FAILED)
        await emit(
            WebSocketEvent(
                event="job_failed",
                project_id=req.project_id,
                job_id=job_id,
                message=f"Lỗi Hardsub OCR: {exc}",
            )
        )
        raise
