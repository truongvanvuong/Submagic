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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

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
from app.services.ffmpeg_service import (
    VideoROIStream,
    crop_and_mask_roi,
    extract_video_pts_table,
    get_timestamp_for_frame,
    probe_video,
    resolve_media_path,
)
from app.services.ocr_engines import (
    clean_ocr_text,
    detect_text_language,
    is_vietocr_available,
    ocr_engine_cache,
    recover_truncated_text_lines,
    resolve_ocr_engine,
    verify_mask_coverage,
)
from app.utils.cjk_normalizer import (
    convert_s2t,
    convert_t2s,
    count_trad_simp_characters,
    is_mixed_cjk_script,
)
from app.utils.subtitle_formatter import format_subtitles

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]

# Precomputed morphological kernels & percentile fractions to avoid per-frame allocations
_MORPH_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_MORPH_K5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_MORPH_K7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
_TOPHAT_KERNELS: dict[tuple[int, int], np.ndarray] = {}
_PERCENTILE_FRACS = np.array([0.01, 0.50, 0.85, 0.95, 0.995], dtype=np.float64)
_CHROMA_FRACS = np.array([0.50, 0.85, 0.995], dtype=np.float64)


def _get_tophat_kernel(kw: int, kh: int) -> np.ndarray:
    key = (kw, kh)
    kernel = _TOPHAT_KERNELS.get(key)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, key)
        if len(_TOPHAT_KERNELS) < 64:
            _TOPHAT_KERNELS[key] = kernel
    return kernel


def _compute_cdf_percentiles(
    cdf: np.ndarray, total_pixels: float
) -> tuple[int, int, int, int, int]:
    """Compute (p01, p50, p85, p95, p995) in a single vectorized searchsorted call."""
    vals = cdf.searchsorted(_PERCENTILE_FRACS * total_pixels)
    return int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])


def _downscale_crop_for_ocr(img: np.ndarray) -> np.ndarray:
    """Downscale excessively tall crops (>160px) to 120px height while preserving aspect ratio."""
    if img is not None and img.shape[0] > 160:
        sc = 120.0 / float(img.shape[0])
        nw = max(16, int(round(img.shape[1] * sc)))
        return cv2.resize(img, (nw, 120), interpolation=cv2.INTER_AREA)
    return img


def extract_text_binary_mask(
    roi_bgr: np.ndarray,
    luma_threshold: int = 185,
    adaptive: bool = True,
) -> np.ndarray:
    """
    Layer 1 of 3-Layer OpenCV Filter:
    Extract a clean binary mask (0 or 255) of high-contrast subtitle strokes while
    rejecting ~90% of moving background scenery.
    Combines:
    1. Adaptive Contrast-Gated Hybrid Percentile-Otsu thresholding (or fixed luma_threshold fallback).
    2. Morphological Top-Hat for bright text strokes over bright backgrounds.
    3. Canny/gradient stroke-edge adjacency so large uniform bright background regions
       (sky, walls, flashes) without internal character edges are suppressed.
    4. Vectorized Connected-component filtering (NumPy LUT) to remove tiny 1-2px noise specks.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if roi_bgr.ndim == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr

    orig_h, orig_w = gray.shape[:2]
    if orig_h < 4 or orig_w < 4:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Downscale for mask calculation if height > 60 (preserves contours while 4-5x faster)
    if orig_h > 60:
        scale = 48.0 / float(orig_h)
        work_w = max(16, int(round(orig_w * scale)))
        work_gray = cv2.resize(gray, (work_w, 48), interpolation=cv2.INTER_AREA)
    else:
        work_gray = gray

    h, w = work_gray.shape[:2]

    if adaptive:
        hist = cv2.calcHist([work_gray], [0], None, [256], [0, 256]).ravel()
        cdf = hist.cumsum()
        total_pixels = float(cdf[-1]) if len(cdf) > 0 and cdf[-1] > 0 else 1.0

        p01, p50, p85, p95, p995 = _compute_cdf_percentiles(cdf, total_pixels)

        delta_high = (p995 - p50) if p50 >= 140 else (p95 - p50)
        delta_low = p50 - p01
        is_white_bg_outlined = p50 >= 195 and p995 >= 238 and delta_low >= 80
        is_sparse_bright_text = p50 < 140 and p995 >= 135 and (p995 - p50) >= 60

        # Rescue bright high-chroma subtitles (e.g. yellow/cyan text) on bright white/gray backgrounds
        # where standard grayscale luminance contrast against the background is low (< 15).
        if (
            delta_high < 15
            and not is_white_bg_outlined
            and not is_sparse_bright_text
            and roi_bgr.ndim == 3
            and p50 >= 140
        ):
            work_bgr = (
                cv2.resize(roi_bgr, (w, h), interpolation=cv2.INTER_AREA)
                if orig_h > 60
                else roi_bgr
            )
            max_c = np.max(work_bgr, axis=2)
            min_c = np.min(work_bgr, axis=2)
            chroma = cv2.subtract(max_c, min_c)
            c_hist = cv2.calcHist([chroma], [0], None, [256], [0, 256]).ravel()
            c_cdf = c_hist.cumsum()
            c_vals = c_cdf.searchsorted(_CHROMA_FRACS * total_pixels)
            c_p50, c_p85, c_p995 = int(c_vals[0]), int(c_vals[1]), int(c_vals[2])
            if c_p85 <= 45 and (c_p995 - c_p50) >= 60:
                high_chroma = (chroma > (c_p85 + 40)) & (max_c >= 200)
                if np.any(high_chroma):
                    work_gray = work_gray.copy()
                    work_gray[high_chroma] = np.maximum(
                        work_gray[high_chroma], max_c[high_chroma]
                    )
                    hist = cv2.calcHist([work_gray], [0], None, [256], [0, 256]).ravel()
                    cdf = hist.cumsum()
                    p01, p50, p85, p95, p995 = _compute_cdf_percentiles(cdf, total_pixels)
                    delta_high = (p995 - p50) if p50 >= 140 else (p95 - p50)
                    delta_low = p50 - p01
                    is_white_bg_outlined = (
                        p50 >= 195 and p995 >= 238 and delta_low >= 80
                    )

        # Early exit for flat background without text
        if delta_high < 15 and not is_white_bg_outlined and not is_sparse_bright_text:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

        if is_white_bg_outlined:
            # Near-white background (e.g. bright paper/snow/sky) with white text + dark outline
            clamped_thresh = max(234, min(252, p95 + 10))
            relaxed_thresh = max(228, clamped_thresh - 8)
        elif p50 >= 140:
            # Bright scene
            clamped_thresh = max(195, min(245, max(p85 + 8, p50 + 15)))
            relaxed_thresh = max(165, p85 + 4, clamped_thresh - 20)
        else:
            # Dark / mid scene: Otsu threshold with safety bound
            otsu_val, _ = cv2.threshold(work_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if is_sparse_bright_text and delta_high < 15:
                cand = max(p95 + 25, int(round(otsu_val)))
                clamped_thresh = max(125, min(215, cand))
                relaxed_thresh = max(105, clamped_thresh - 20)
            elif (p85 - p50) >= 10:
                # Textured dark/mid scene: enforce threshold above P85 background clutter
                cand = max(p85 + 25, int(round(otsu_val)))
                clamped_thresh = max(145, min(215, cand))
                relaxed_thresh = max(125, clamped_thresh - 20)
            else:
                # Uniform dark background
                cand = max(p50 + 20, int(round(otsu_val)))
                clamped_thresh = max(65, min(210, cand))
                relaxed_thresh = max(45, clamped_thresh - 20)
    else:
        is_white_bg_outlined = False
        clamped_thresh = max(50, min(254, int(luma_threshold)))
        relaxed_thresh = max(135, clamped_thresh - 30)

    # Fast early-exit: if there are fewer than 6 bright pixels, no subtitle text can form
    _, relaxed_bright = cv2.threshold(work_gray, relaxed_thresh, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(relaxed_bright) < 6:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # 1. Direct luminance threshold for bright subtitle characters
    _, bright_mask = cv2.threshold(work_gray, clamped_thresh, 255, cv2.THRESH_BINARY)

    # 2. Top-Hat transform to isolate bright thin strokes over moderately bright backgrounds
    kw = max(9, min(31, (w // 20) | 1))
    kh = max(7, min(21, (h // 4) | 1))
    tophat_kernel = _get_tophat_kernel(kw, kh)
    tophat = cv2.morphologyEx(work_gray, cv2.MORPH_TOPHAT, tophat_kernel)
    tophat_cutoff = max(30, min(90, (255 - clamped_thresh) // 2 + 15))
    _, tophat_mask = cv2.threshold(tophat, tophat_cutoff, 255, cv2.THRESH_BINARY)

    # 3. Canny edge detection around character strokes
    edges = cv2.Canny(work_gray, 50, 150)
    edge_zone = cv2.dilate(edges, _MORPH_K5, iterations=1)

    if is_white_bg_outlined:
        # On near-white backgrounds, require bright character pixels to sit adjacent to both
        # Canny edges and dark character outlines so bright background patches are rejected
        _, dark_outline = cv2.threshold(work_gray, min(140, p50 - 65), 255, cv2.THRESH_BINARY_INV)
        outline_zone = cv2.dilate(dark_outline, _MORPH_K7, iterations=1)
        combined = cv2.bitwise_and(cv2.bitwise_and(bright_mask, edge_zone), outline_zone)
    else:
        # Bright pixels must sit near character edges (eliminates interior of large bright background patches)
        stroke_candidates = cv2.bitwise_and(bright_mask, edge_zone)
        tophat_bright = cv2.bitwise_and(bright_mask, tophat_mask)
        tophat_edges = cv2.bitwise_and(
            cv2.bitwise_and(relaxed_bright, tophat_mask), edge_zone
        )
        combined = cv2.bitwise_or(
            stroke_candidates, cv2.bitwise_or(tophat_bright, tophat_edges)
        )

    # Close tiny 1-2px gaps inside character strokes
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _MORPH_K3, iterations=1)

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

    if orig_h > 60:
        return cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return clean_mask


def _has_text_activity_from_count(
    mask: np.ndarray,
    nz: int,
    min_pixels: int = 12,
    min_ratio: float = 0.001,
    max_ratio: float = 0.85,
    verify_dense_strokes: bool = True,
) -> bool:
    if nz < min_pixels:
        return False
    ratio = nz / float(mask.size)
    if ratio < min_ratio or ratio > max_ratio:
        return False

    # Tier 1: Fast-path for normal subtitle densities <= 0.65
    if ratio <= 0.65 or not verify_dense_strokes:
        return True

    # Tier 2: Verified Stroke Filter for high densities (0.65 < ratio <= 0.85)
    # Genuine text strokes (width >= 2-3px) preserve an inner core when eroded by 3x3.
    # Fragmented noise / texture clutter evaporates (< 10-12% core remaining).
    eroded = cv2.erode(mask, _MORPH_K3, iterations=1)
    nz_core = int(cv2.countNonZero(eroded))
    if nz_core == 0:
        return False
    return (nz_core / float(nz)) >= 0.12


def has_text_activity(
    mask: np.ndarray,
    min_pixels: int = 12,
    min_ratio: float = 0.001,
    max_ratio: float = 0.85,
    verify_dense_strokes: bool = True,
) -> bool:
    """Return True if the binary mask contains plausible subtitle text strokes.

    Accepts density in [0.001, 0.65] unconditionally (fast-path).
    Expands density up to 0.85 only for verified stroke edges (retaining >= 12% core).
    """
    if mask is None or mask.size == 0:
        return False
    nz = int(cv2.countNonZero(mask))
    return _has_text_activity_from_count(
        mask,
        nz,
        min_pixels=min_pixels,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        verify_dense_strokes=verify_dense_strokes,
    )


def compute_mask_visual_diff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Layer 3 of 3-Layer OpenCV Filter:
    Compute visual dissimilarity in [0.0, 1.0] between two binary text masks.
    Uses fast early-rejection on pixel count followed by 1-pixel dilated Dice similarity.
    """
    if mask_a is None or mask_b is None or mask_a.size == 0 or mask_b.size == 0:
        return 1.0

    count_a = int(cv2.countNonZero(mask_a))
    count_b = int(cv2.countNonZero(mask_b))
    active_a = _has_text_activity_from_count(mask_a, count_a)
    active_b = _has_text_activity_from_count(mask_b, count_b)

    if not active_a and not active_b:
        return 0.0
    if active_a != active_b:
        return 1.0

    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(
            mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        count_b = int(cv2.countNonZero(mask_b))

    total_fg = count_a + count_b
    if total_fg == 0:
        return 0.0

    size_ratio = min(count_a, count_b) / max(count_a, count_b)
    # Early-exit: If subtitle pixel areas differ by > 35%, they are definitely different text lines
    if size_ratio < 0.65:
        return 1.0

    # Downscale for comparison only if very large (preserves thin strokes while 10x faster)
    h, w = mask_a.shape[:2]
    if h > 120:
        scale = 96.0 / float(h)
        dw = max(16, int(round(w * scale)))
        m_a = cv2.resize(mask_a, (dw, 96), interpolation=cv2.INTER_NEAREST)
        m_b = cv2.resize(mask_b, (dw, 96), interpolation=cv2.INTER_NEAREST)
        cm_a = int(cv2.countNonZero(m_a))
        cm_b = int(cv2.countNonZero(m_b))
    else:
        m_a, m_b = mask_a, mask_b
        cm_a, cm_b = count_a, count_b

    total_m = cm_a + cm_b
    if total_m == 0:
        return 0.0

    dil_b = cv2.dilate(m_b, _MORPH_K3, iterations=1)
    overlap = cv2.countNonZero(cv2.bitwise_and(m_a, dil_b))
    tolerant_dice = min(1.0, (2.0 * overlap) / float(total_m))

    similarity = tolerant_dice * (0.75 + 0.25 * size_ratio)
    diff = 1.0 - max(0.0, min(1.0, similarity))

    # Detect partial word/character changes and geometry shifts on direct cuts
    dil_a = cv2.dilate(m_a, _MORPH_K3, iterations=1)
    uncovered_a = cv2.countNonZero(cv2.subtract(m_a, dil_b))
    uncovered_b = cv2.countNonZero(cv2.subtract(m_b, dil_a))
    u_change = (uncovered_a + uncovered_b) / float(max(1, min(cm_a, cm_b)))

    # Horizontal geometry shift (width / right edge difference) using direct boundingRect on mask
    xa, _, wa, _ = cv2.boundingRect(m_a)
    xb, _, wb, _ = cv2.boundingRect(m_b)
    if wa > 0 and wb > 0:
        max_w = max(wa, wb, 1)
        geom_diff = (abs(wa - wb) + abs((xa + wa) - (xb + wb))) / float(max_w * 2)
    else:
        geom_diff = 0.0

    total_diff = max(diff, round(u_change + geom_diff, 4))
    clamped_diff = min(1.0, max(0.0, total_diff))
    return round(clamped_diff, 4)


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
    h, w = gray.shape[:2]
    if h > 80:
        scale = 64.0 / float(h)
        dw = max(16, int(round(w * scale)))
        small_gray = cv2.resize(gray, (dw, 64), interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray

    lap_var = float(cv2.Laplacian(small_gray, cv2.CV_32F).var())
    if mask is not None and mask.shape == gray.shape:
        nz = cv2.countNonZero(mask)
        if nz > 0:
            fg_luma = float(cv2.mean(gray, mask=mask)[0])
            return lap_var + (fg_luma * 2.5) + min(200.0, float(nz) * 0.05)
    return lap_var


def compute_min_pooled_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    """
    Combine 3..8 frames of a subtitle span using pixel-wise minimum.
    Suppresses moving/flashing bright background regions while keeping bright text strokes.
    Limitation: assumes bright text on a darker/varying background; may degrade dark text
    or fading subtitles, so callers should always compare against the best single frame.
    """
    if len(frames) < 3:
        return None
    ref_shape = frames[0].shape
    compatible = [f for f in frames[:8] if f is not None and f.shape == ref_shape]
    if len(compatible) < 3:
        return None
    return np.min(np.stack(compatible, axis=0), axis=0)


def select_voted_ocr_result(
    candidates: list[tuple[OCRFrameResult, np.ndarray]],
) -> tuple[OCRFrameResult, np.ndarray]:
    """
    Vote across multiple OCR results from the same subtitle span (best frame, min-pooled frame,
    and candidate frames). Groups by normalized text, preferring majority vote, then total
    confidence, and breaking near-ties among high-confidence results in favor of longer text
    (which recovers characters dropped by CTC decoding).
    """
    non_empty = [(r, img) for r, img in candidates if clean_ocr_text(r.text)]
    if not non_empty:
        return candidates[0]

    buckets: dict[str, list[tuple[OCRFrameResult, np.ndarray]]] = {}
    for r, img in non_empty:
        key = convert_t2s(clean_ocr_text(r.text)).lower()
        buckets.setdefault(key, []).append((r, img))

    def _bucket_score(items: list[tuple[OCRFrameResult, np.ndarray]]) -> tuple[int, float]:
        count = len(items)
        avg_conf = sum(float(r.confidence or 0.0) for r, _ in items) / count
        return (count, avg_conf)

    best_key = max(buckets.keys(), key=lambda k: _bucket_score(buckets[k]))
    best_group = buckets[best_key]
    return max(best_group, key=lambda pair: float(pair[0].confidence or 0.0))


@dataclass
class VisualSubtitleSpan:
    """Represents a stabilized visual subtitle appearance interval ready for AI OCR."""
    start_time: float
    end_time: float
    best_frame_bgr: np.ndarray
    best_mask: np.ndarray
    best_sharpness: float = 0.0
    sampled_frames: int = 1
    candidate_frames: list[np.ndarray] = field(default_factory=list)
    is_timeout_split: bool = False


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
        diagnostic_logging: bool = False,
        adaptive_threshold: bool = True,
        max_stroke_density: float = 0.85,
    ) -> None:
        self.luma_threshold = luma_threshold
        self.stability_frames = max(0, int(stability_frames))
        self.visual_diff_threshold = float(visual_diff_threshold)
        self.frame_step_sec = max(0.02, float(frame_step_sec))
        self.diagnostic_logging = bool(diagnostic_logging)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.max_stroke_density = float(max_stroke_density)

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
            else extract_text_binary_mask(
                roi_bgr,
                luma_threshold=self.luma_threshold,
                adaptive=self.adaptive_threshold,
            )
        )
        has_text = has_text_activity(mask, max_ratio=self.max_stroke_density)
        if has_text:
            self.frames_with_text += 1

        completed: list[VisualSubtitleSpan] = []

        if not has_text:
            if self.diagnostic_logging:
                logger.debug(
                    "[OCR DIAG] Frame at %.3fs: dropped candidate text due to low activity (has_text_activity=False)",
                    t,
                )
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
            elif self._state == "STABILIZING" and self._pending_span is not None:
                if self.diagnostic_logging:
                    logger.info(
                        "[OCR DIAG] Span [%.3fs -> %.3fs]: dropped candidate due to stability filter (sampled_frames=%d <= stability_frames=%d)",
                        self._pending_span.start_time,
                        self._pending_span.end_time,
                        self._pending_span.sampled_frames,
                        self.stability_frames,
                    )
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

            if diff < self.visual_diff_threshold:
                # Mask is consistent with previous frame
                self._pending_span.end_time = next_t
                self._pending_span.sampled_frames += 1
                self._stable_count += 1
                if sharpness >= self._pending_span.best_sharpness:
                    frame_copy = roi_bgr.copy()
                    self._pending_span.best_frame_bgr = frame_copy
                    self._pending_span.best_mask = mask
                    self._pending_span.best_sharpness = sharpness
                    if len(self._pending_span.candidate_frames) < 8:
                        self._pending_span.candidate_frames.append(frame_copy)
                elif len(self._pending_span.candidate_frames) < 8:
                    self._pending_span.candidate_frames.append(roi_bgr.copy())
                self._pending_last_mask = mask

                if self._stable_count >= self.stability_frames:
                    # Promoted to ACTIVE!
                    self._state = "ACTIVE"
                    self._active_span = self._pending_span
                    self._pending_span = None
                    self._pending_last_mask = None
                    self.ocr_triggers_count += 1
            else:
                dil_inter = cv2.countNonZero(
                    cv2.bitwise_and(
                        self._pending_last_mask, cv2.dilate(mask, _MORPH_K3, iterations=1)
                    )
                )
                min_px = min(
                    cv2.countNonZero(self._pending_last_mask), cv2.countNonZero(mask)
                )
                raw_ov = dil_inter / float(max(1, min_px))
                # Case A: Previous pending frames were a weak background transient and a much sharper
                # subtitle just appeared at a different location -> start fresh at t (do not shift start_time early!)
                if raw_ov < 0.45 and sharpness > self._pending_span.best_sharpness * 1.25:
                    if (
                        self._pending_span.sampled_frames > self.stability_frames
                        and self._pending_span.best_sharpness >= 450.0
                    ):
                        self._pending_span.end_time = max(
                            round(self._pending_span.start_time + 0.15, 3),
                            min(self._pending_span.end_time, t),
                        )
                        self.ocr_triggers_count += 1
                        completed.append(self._pending_span)
                    self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)
                    return completed

                # Case B: Previous pending span was a sharp subtitle and current frame dropped in
                # sharpness at a disjoint location (subtitle disappeared into background residue) -> end at t!
                if raw_ov < 0.45 and sharpness < self._pending_span.best_sharpness * 0.75:
                    if self._pending_span.sampled_frames > self.stability_frames:
                        self._pending_span.end_time = max(
                            round(self._pending_span.start_time + 0.15, 3),
                            min(self._pending_span.end_time, t),
                        )
                        self.ocr_triggers_count += 1
                        completed.append(self._pending_span)
                    self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)
                    return completed

                # Case C: Still fading in or changing shape across > 0.65s (e.g. karaoke build-up or fast cut)
                if (
                    (t - self._pending_span.start_time) > 0.65
                    and self._pending_span.sampled_frames > self.stability_frames
                ):
                    self._pending_span.end_time = max(
                        round(self._pending_span.start_time + 0.15, 3), t
                    )
                    self._pending_span.is_timeout_split = True
                    self.ocr_triggers_count += 1
                    completed.append(self._pending_span)
                    self._start_new_candidate(roi_bgr, mask, sharpness, t, next_t)
                    return completed

                self._pending_span.end_time = next_t
                self._pending_span.sampled_frames += 1
                if sharpness >= self._pending_span.best_sharpness:
                    frame_copy = roi_bgr.copy()
                    self._pending_span.best_frame_bgr = frame_copy
                    self._pending_span.best_mask = mask
                    self._pending_span.best_sharpness = sharpness
                    if len(self._pending_span.candidate_frames) < 8:
                        self._pending_span.candidate_frames.append(frame_copy)
                elif len(self._pending_span.candidate_frames) < 8:
                    self._pending_span.candidate_frames.append(roi_bgr.copy())
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
                    frame_copy = roi_bgr.copy()
                    self._active_span.best_frame_bgr = frame_copy
                    self._active_span.best_mask = mask
                    self._active_span.best_sharpness = sharpness
                    if len(self._active_span.candidate_frames) < 8:
                        self._active_span.candidate_frames.append(frame_copy)
                elif len(self._active_span.candidate_frames) < 8:
                    self._active_span.candidate_frames.append(roi_bgr.copy())
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
        frame_copy = roi_bgr.copy()
        span = VisualSubtitleSpan(
            start_time=start_t,
            end_time=end_t,
            best_frame_bgr=frame_copy,
            best_mask=mask,
            best_sharpness=sharpness,
            sampled_frames=1,
            candidate_frames=[frame_copy],
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


CJK_CHAR_REGEX = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)
CJK_PUNCTUATION = set("，。！？、；：""''（）《》〈〉【】…—～·.,!?;:\"'()[]")


def is_cjk_string(text: str) -> bool:
    """Return True if text contains at least one CJK ideograph, kana, or hangul character."""
    if not text:
        return False
    return bool(CJK_CHAR_REGEX.search(text))


def extract_meaningful_cjk_characters(text: str) -> list[str]:
    """Extract semantic CJK characters, ignoring whitespace and punctuation."""
    return [
        ch for ch in text
        if not ch.isspace() and ch not in CJK_PUNCTUATION and ch != "\u3000"
    ]


_CJK_NUMERIC_UNITS = set("年月日天个個次倍万萬亿億分秒点點岁歲米块塊元号號名人种種层層楼樓章集回话話")


def clean_cjk_subtitle_text(text: str) -> str:
    """
    Remove common background OCR edge artifacts from CJK subtitle lines without touching
    non-CJK subtitles:
    - Leading/trailing isolated ASCII/digit/symbol noise (e.g. "5死了", "1你没死", "啊 DDN",
      "你們了星 S", "4 个你很失望", "L 雀他是怎做到的", "盈盈B"), while preserving numeric
      expressions like "2024年", "50万", "3天".
    - Space-separated trailing 1-char duplicate CJK (e.g. "令狐公子 子" -> "令狐公子").
    - Spurious spaces inserted between adjacent CJK characters (e.g. "别 來 無 恙 啊").
    """
    if not text or not is_cjk_string(text):
        return text

    s = text.strip()
    # 1. Strip space-separated leading short ASCII/symbol noise before CJK (e.g. "4 个...", "L 雀...", "A 扌據...")
    s = re.sub(r"^(?:[A-Za-z0-9↑↓←→=•·|{}[\]()<>*#]{1,4}\s+)+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])", "", s)
    # 2. Strip glued leading 1-2 char ASCII/digit/symbol noise before CJK unless followed by a numeric unit
    m_lead = re.match(r"^([A-Za-z0-9↑↓←→=•·|{}[\]()<>*#]{1,2})([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])", s)
    if m_lead:
        prefix, next_ch = m_lead.group(1), m_lead.group(2)
        if not (prefix.isdigit() and next_ch in _CJK_NUMERIC_UNITS):
            s = s[len(prefix):]
    # 3. Strip space-separated trailing short ASCII/digit/symbol/katakana noise (e.g. "啊 5 クン", "啊 DDN", "你們了星 S")
    s = re.sub(r"(?:\s+[A-Za-z0-9↑↓←→=•·|{}[\]()<>*#]{1,5})+(?:\s+[\u30a0-\u30ff]{1,3})?$", "", s)
    # Also strip trailing glued 1-2 Katakana noise after Han ideograph when preceded by space (e.g. "啊 クン")
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+[\u30a0-\u30ff]{1,2}$", "", s)
    # 4. Strip single glued trailing uppercase/digit/symbol after >= 2 CJK chars (e.g. "盈盈B", "你們了S")
    s = re.sub(r"(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2})[A-Z0-9=↑↓←→•·|{}[\]()<>*#]$", "", s)
    # 5. Strip space-separated trailing 1-char duplicate CJK (e.g. "令狐公子 子" -> "令狐公子")
    s = re.sub(r"([\u4e00-\u9fff])\s+\1$", r"\1", s)
    # 6. Collapse spurious spaces between CJK characters
    s = re.sub(
        r"(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])",
        "",
        s,
    )
    return s.strip()


def filter_cjk_majority_noise_segments(
    segments: list[SubtitleSegment],
    language: str,
) -> tuple[list[SubtitleSegment], list[SubtitleSegment]]:
    """
    When the target language is CJK or auto-detected majority (>= 60% of >= 5 segments) is CJK,
    filter out pure non-CJK hallucinations (e.g. "YA", "WA", "ay", "D", "9", "JY") and
    predominantly-Latin noise fragments containing <= 2 CJK chars (e.g. "ULUDE 茶 喉", "文 JT").
    """
    if not segments:
        return segments, []
    lang = (language or "auto").strip().lower()
    cjk_count = sum(1 for seg in segments if is_cjk_string(seg.text))
    is_cjk_video = lang in ("ch", "ja", "ko") or (
        lang == "auto" and len(segments) >= 5 and (cjk_count / len(segments)) >= 0.60
    )
    if not is_cjk_video:
        return segments, []

    kept: list[SubtitleSegment] = []
    dropped: list[SubtitleSegment] = []
    for seg in segments:
        cjk_core = extract_cjk_core(seg.text)
        ascii_letters = sum(1 for ch in seg.text if ch.isascii() and ch.isalpha())
        if not cjk_core or (len(cjk_core) <= 2 and ascii_letters > len(cjk_core)):
            dropped.append(seg)
        else:
            kept.append(seg)
    for idx, seg in enumerate(kept, start=1):
        seg.index = idx
    return kept, dropped


def extract_cjk_core(text: str) -> str:
    """
    Return only the CJK ideographs/kana/hangul of `text` (Traditional folded to Simplified) as a string.
    Latin letters, digits, punctuation and whitespace (typical background OCR noise such as
    "A", "1", "L Y", "S") are dropped so fragments of the same subtitle compare equal.
    """
    if not text:
        return ""
    folded = convert_t2s(text)
    return "".join(ch for ch in folded if CJK_CHAR_REGEX.match(ch))


def is_cjk_subsequence(short: str, long: str) -> bool:
    """True if every character of `short` appears in `long` in the same order."""
    it = iter(long)
    return all(ch in it for ch in short)


def is_same_cjk_subtitle_fragment(text_a: str, text_b: str, max_len_diff: int = 2) -> bool:
    """
    Decide whether two adjacent OCR fragments are the same CJK subtitle line that was split
    by background motion or the 0.65s stabilizing timeout. Compares CJK cores only:
    - identical cores (>= 2 chars),
    - ordered subsequence with length difference <= `max_len_diff` (e.g. "真点" vs "言真点"), or
    - high LCS overlap (>= 75% and >= 3 shared chars) where differing chars are edge/noise
      artifacts rather than an interior word change between two distinct sentences.
    """
    core_a = extract_cjk_core(text_a)
    core_b = extract_cjk_core(text_b)
    if not core_a or not core_b:
        return False
    if core_a == core_b:
        return len(core_a) >= 2
    short, long = (core_a, core_b) if len(core_a) <= len(core_b) else (core_b, core_a)
    len_diff = len(long) - len(short)
    if len_diff <= max_len_diff:
        # Short 1-2 char CJK lines with 1-2 leading/trailing noise chars (e.g. "你" vs "个你", "怎麼" vs "一八怎麼")
        if len(short) >= 1 and (long.startswith(short) or long.endswith(short)):
            return True
        min_sub_len = 2 if len_diff == 1 else 3
        if len(short) >= min_sub_len and is_cjk_subsequence(short, long):
            return True

    # Handle bilateral noise insertions (e.g. "女如你所" vs "如口你所", "希望我是主" vs "吴希望我是",
    # "雀他是怎做到的" vs "个他是怎做到的", "能奈我我何" vs "肩能奈我何", "令狐公子二" vs "今狐公子")
    if len_diff <= max_len_diff and len(short) >= 3:
        import difflib

        sm = difflib.SequenceMatcher(None, core_a, core_b)
        lcs = sum(block.size for block in sm.get_matching_blocks())
        ratio = (2.0 * lcs) / (len(core_a) + len(core_b))
        longest_block = max((b.size for b in sm.get_matching_blocks()), default=0)
        if lcs >= 3 and (ratio >= 0.75 or (longest_block >= 3 and ratio >= 0.66 and len_diff <= 1)):
            # Guard: if two >= 7-char sentences of equal length differ by an interior substitution,
            # keep them separate (distinct sentences like "我今天要去北京" vs "我明天要去北京").
            if len(core_a) == len(core_b) >= 7 and lcs == len(core_a) - 1:
                if core_a[1:] == core_b[1:] or core_a[:-1] == core_b[:-1]:
                    return True
                return False
            return True
    return False


def _strip_diacritics_alnum(text: str) -> str:
    """Fold Unicode diacritics and Vietnamese đ->d, keeping only lowercase ASCII/alphanumeric chars."""
    import unicodedata

    folded = text.lower().replace("đ", "d")
    nfd = unicodedata.normalize("NFD", folded)
    return "".join(
        ch for ch in nfd if unicodedata.category(ch) != "Mn" and ch.isalnum()
    )


def is_same_subtitle_fragment(text_a: str, text_b: str) -> bool:
    """
    Language-agnostic check for whether two consecutive OCR fragments belong to the same
    on-screen subtitle line that was split by background motion or fade/timeout.
    Works across CJK, Vietnamese, English, and other alphabetic scripts.
    """
    if is_same_cjk_subtitle_fragment(text_a, text_b):
        return True
    if is_cjk_string(text_a) or is_cjk_string(text_b):
        return False

    norm_a = normalize_for_fuzzy(text_a)
    norm_b = normalize_for_fuzzy(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return len(norm_a) >= 2

    core_a = "".join(ch for ch in norm_a if ch.isalnum())
    core_b = "".join(ch for ch in norm_b if ch.isalnum())
    if not core_a or not core_b:
        return False
    if core_a == core_b:
        return len(core_a) >= 2

    # Guard against merging sequential/distinct multi-word sentences that share a template
    # (e.g. "Cảnh quay thứ nhất" vs "Cảnh quay thứ hai", "Part one" vs "Part two")
    words_a = [_strip_diacritics_alnum(w) for w in norm_a.split() if _strip_diacritics_alnum(w)]
    words_b = [_strip_diacritics_alnum(w) for w in norm_b.split() if _strip_diacritics_alnum(w)]
    if len(words_a) >= 2 and len(words_b) >= 2 and len(words_a) == len(words_b):
        if (
            len(words_a[-1]) >= 2
            and len(words_b[-1]) >= 2
            and float(fuzz.ratio(words_a[-1], words_b[-1])) < 65.0
        ):
            return False
        if (
            len(words_a[0]) >= 2
            and len(words_b[0]) >= 2
            and float(fuzz.ratio(words_a[0], words_b[0])) < 65.0
        ):
            return False

    short_c, long_c = (
        (core_a, core_b) if len(core_a) <= len(core_b) else (core_b, core_a)
    )
    len_diff = len(long_c) - len(short_c)

    # Prefix/suffix or substring with up to 3 leading/trailing OCR noise characters
    if len(short_c) >= 4 and len_diff <= 3 and short_c in long_c:
        return True

    if compute_ocr_text_similarity(text_a, text_b) >= 82.0:
        return True

    if len(short_c) >= 4 and float(fuzz.ratio(core_a, core_b)) >= 84.0:
        return True

    # Diacritic-insensitive alphanumeric comparison for accented scripts (e.g. Vietnamese OCR jitter)
    ascii_a = _strip_diacritics_alnum(text_a)
    ascii_b = _strip_diacritics_alnum(text_b)
    if (
        len(ascii_a) >= 5
        and len(ascii_b) >= 5
        and (ascii_a[:2] == ascii_b[:2] or ascii_a[-2:] == ascii_b[-2:])
        and float(fuzz.ratio(ascii_a, ascii_b)) >= 78.0
    ):
        return True

    return False


def add_fragment_vote(
    votes: dict[str, dict[str, Any]],
    text: str,
    duration: float,
    confidence: float,
    original_text: str | None = None,
    normalized_text: str | None = None,
    crop_image_url: str | None = None,
) -> None:
    """Accumulate a duration-weighted vote for one OCR reading of a (split) CJK subtitle line."""
    core = extract_cjk_core(text) or normalize_for_fuzzy(text)
    entry = votes.get(core)
    weight = max(0.01, float(duration))
    if entry is None:
        votes[core] = {
            "weight": weight,
            "text": text,
            "confidence": float(confidence),
            "original_text": original_text,
            "normalized_text": normalized_text,
            "crop_image_url": crop_image_url,
        }
        return
    entry["weight"] += weight
    # Within the same CJK core, keep the representation with the highest confidence
    if float(confidence) > entry["confidence"]:
        entry.update(
            text=text,
            confidence=float(confidence),
            original_text=original_text,
            normalized_text=normalized_text,
            crop_image_url=crop_image_url or entry["crop_image_url"],
        )


def _cjk_noise_penalty(text: str) -> int:
    """Count non-CJK ASCII/digit/space noise characters inside a CJK subtitle reading."""
    if not is_cjk_string(text):
        return 0
    return sum(1 for ch in text if ch.isspace() or (ch.isascii() and ch.isalnum()))


def select_fragment_vote(votes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick the reading with the largest on-screen duration; prefer cleaner text on close ties."""
    return max(
        votes.values(),
        key=lambda e: (round(e["weight"], 2), -_cjk_noise_penalty(e["text"]), e["confidence"]),
    )


def tokenize_for_dedup(text: str) -> list[str]:
    """
    Tokenize text for deduplication SequenceMatcher:
    - Non-CJK words (delimited by spaces) remain words.
    - CJK ideograms are individual single-character tokens.
    - Punctuation characters are stripped so they do not form false word tokens.
    """
    tokens: list[str] = []
    for word in text.split():
        curr_latin: list[str] = []
        for ch in word:
            if ch in CJK_PUNCTUATION:
                if curr_latin:
                    tokens.append("".join(curr_latin))
                    curr_latin = []
            elif CJK_CHAR_REGEX.match(ch):
                if curr_latin:
                    tokens.append("".join(curr_latin))
                    curr_latin = []
                tokens.append(ch)
            else:
                curr_latin.append(ch)
        if curr_latin:
            tokens.append("".join(curr_latin))
    return tokens


class FuzzySubtitleDeduplicator:
    """
    Merges consecutive OCR subtitle segments whose normalized Levenshtein similarity
    (`rapidfuzz.fuzz.ratio`) is >= `fuzzy_match_threshold` (default 85%) and whose
    time gap is <= `max_merge_gap_sec` (default 0.20s).
    Automatically retains the text and `crop_image_url` of the higher-confidence recognition.

    Guards:
    1. Feature 16: Mask Visual Difference Guard blocks merging when binary mask difference >= threshold.
    2. Feature 15: CJK Character-Level Guard blocks merging for CJK strings of length >= 7 differing by any character.
    3. Short Subtitle Guard: Subtitles <= 4 chars require 100% exact match.
    4. SequenceMatcher Guard: Replaced words/tokens with similarity < 60% block merging.
    """

    def __init__(
        self,
        fuzzy_match_threshold: float = 85.0,
        max_merge_gap_sec: float = 0.20,
        mask_visual_diff_threshold: float = 0.25,
    ) -> None:
        self.fuzzy_match_threshold = float(fuzzy_match_threshold)
        self.max_merge_gap_sec = float(max_merge_gap_sec)
        self.mask_visual_diff_threshold = float(mask_visual_diff_threshold)
        self.segments: list[SubtitleSegment] = []
        self._last_mask: np.ndarray | None = None

    def add_or_merge(
        self,
        candidate: SubtitleSegment,
        mask: np.ndarray | None = None,
    ) -> tuple[SubtitleSegment, bool]:
        """
        Add `candidate` segment or merge it into the previous segment if similar & adjacent.
        Returns `(segment, was_merged)`.
        """
        if not self.segments:
            candidate.index = 1
            self.segments.append(candidate)
            self._last_mask = mask
            return candidate, False

        prev = self.segments[-1]
        gap = max(0.0, candidate.start_time - prev.end_time)

        # Gap check: if gap exceeds max_merge_gap_sec, do not merge
        if gap > self.max_merge_gap_sec:
            candidate.index = len(self.segments) + 1
            self.segments.append(candidate)
            self._last_mask = mask
            return candidate, False

        # Feature 16: Mask Visual Difference Guard
        # If visual dissimilarity between masks exceeds threshold, strictly forbid merging
        if self._last_mask is not None and mask is not None:
            mask_diff = compute_mask_visual_diff(self._last_mask, mask)
            if mask_diff >= self.mask_visual_diff_threshold:
                candidate.index = len(self.segments) + 1
                self.segments.append(candidate)
                self._last_mask = mask
                return candidate, False

        norm_p = normalize_for_fuzzy(prev.normalized_text or convert_t2s(prev.text))
        norm_c = normalize_for_fuzzy(candidate.normalized_text or convert_t2s(candidate.text))

        # Feature 15: CJK Character-Level Guard & Semantic Equality Merge
        if is_cjk_string(norm_p) and is_cjk_string(norm_c):
            chars_p = extract_meaningful_cjk_characters(norm_p)
            chars_c = extract_meaningful_cjk_characters(norm_c)

            # CJK Semantic Equality: exact same meaningful characters -> merge directly
            if chars_p == chars_c and len(chars_p) > 0:
                prev.start_time = round(min(prev.start_time, candidate.start_time), 3)
                prev.end_time = round(max(prev.end_time, candidate.end_time), 3)

                # Select best text representation: prioritize richer punctuation, then higher confidence
                punct_p = sum(1 for ch in prev.text if ch in CJK_PUNCTUATION)
                punct_c = sum(1 for ch in candidate.text if ch in CJK_PUNCTUATION)

                cand_better = (
                    (punct_c > punct_p)
                    or (
                        punct_c == punct_p
                        and (
                            candidate.confidence > prev.confidence
                            or (
                                abs(candidate.confidence - prev.confidence) < 1e-4
                                and len(candidate.text) > len(prev.text)
                            )
                        )
                    )
                )

                if cand_better:
                    prev.text = candidate.text
                    prev.original_text = candidate.original_text or candidate.text
                    prev.normalized_text = candidate.normalized_text or convert_t2s(candidate.text)
                    if candidate.crop_image_url:
                        prev.crop_image_url = candidate.crop_image_url
                else:
                    if not prev.original_text:
                        prev.original_text = prev.text
                    if not prev.normalized_text:
                        prev.normalized_text = convert_t2s(prev.text)

                prev.is_low_confidence = prev.is_low_confidence or candidate.is_low_confidence
                prev.confidence = round(max(prev.confidence, candidate.confidence), 4)
                if mask is not None:
                    self._last_mask = mask

                return prev, True

            # If differing and length >= 7, strictly block merging (R3 requirement)
            if max(len(chars_p), len(chars_c)) >= 7:
                candidate.index = len(self.segments) + 1
                self.segments.append(candidate)
                self._last_mask = mask
                return candidate, False

        similarity = compute_ocr_text_similarity(prev.text, candidate.text)

        # Short subtitles (<= 4 characters, e.g. Chinese 2-char words) require 100% exact match
        min_sim = (
            100.0
            if (len(norm_p) <= 4 or len(norm_c) <= 4)
            else self.fuzzy_match_threshold
        )

        if similarity < min_sim:
            candidate.index = len(self.segments) + 1
            self.segments.append(candidate)
            self._last_mask = mask
            return candidate, False

        # Token-based SequenceMatcher Guard (works for both CJK characters and Latin/Vietnamese words)
        tokens_p = tokenize_for_dedup(norm_p)
        tokens_c = tokenize_for_dedup(norm_c)
        if len(tokens_p) > 1 or len(tokens_c) > 1:
            import difflib

            sm = difflib.SequenceMatcher(None, tokens_p, tokens_c)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "replace":
                    for wp, wc in zip(tokens_p[i1:i2], tokens_c[j1:j2]):
                        if wp != wc and fuzz.ratio(wp, wc) < 60.0:
                            candidate.index = len(self.segments) + 1
                            self.segments.append(candidate)
                            self._last_mask = mask
                            return candidate, False
                    if (i2 - i1) != (j2 - j1):
                        candidate.index = len(self.segments) + 1
                        self.segments.append(candidate)
                        self._last_mask = mask
                        return candidate, False
                elif tag in ("insert", "delete"):
                    diff_words = tokens_c[j1:j2] if tag == "insert" else tokens_p[i1:i2]
                    for dw in diff_words:
                        min_len = 1 if is_cjk_string(dw) else 2
                        if len(dw) >= min_len:
                            candidate.index = len(self.segments) + 1
                            self.segments.append(candidate)
                            self._last_mask = mask
                            return candidate, False

        prev.start_time = round(min(prev.start_time, candidate.start_time), 3)
        prev.end_time = round(max(prev.end_time, candidate.end_time), 3)

        # Pick the text & crop image with higher confidence (or longer text on tie)
        cand_better = (candidate.confidence > prev.confidence) or (
            abs(candidate.confidence - prev.confidence) < 1e-4
            and len(candidate.text) > len(prev.text)
        )
        if cand_better:
            prev.text = candidate.text
            prev.original_text = candidate.original_text or candidate.text
            prev.normalized_text = candidate.normalized_text or convert_t2s(candidate.text)
            prev.confidence = round(candidate.confidence, 4)
            if candidate.crop_image_url:
                prev.crop_image_url = candidate.crop_image_url
        else:
            if not prev.original_text:
                prev.original_text = prev.text
            if not prev.normalized_text:
                prev.normalized_text = convert_t2s(prev.text)

        prev.is_low_confidence = prev.is_low_confidence or candidate.is_low_confidence
        if mask is not None:
            self._last_mask = mask

        return prev, True


def deduplicate_ocr_segments(
    segments: list[SubtitleSegment],
    fuzzy_match_threshold: float = 85.0,
    max_merge_gap_sec: float = 0.20,
    mask_visual_diff_threshold: float = 0.25,
) -> list[SubtitleSegment]:
    """
    Standalone helper to deduplicate and merge a list of OCR `SubtitleSegment` items
    using RapidFuzz Levenshtein similarity and time-gap constraints.
    """
    dedup = FuzzySubtitleDeduplicator(
        fuzzy_match_threshold=fuzzy_match_threshold,
        max_merge_gap_sec=max_merge_gap_sec,
        mask_visual_diff_threshold=mask_visual_diff_threshold,
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
    quality: int = 85,
) -> str | None:
    """
    Save the cropped ROI BGR frame to `data/crops/<project_id>/` and return its
    static URL (`/static/crops/<project_id>/<filename>`) for the interactive subtitle editor.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    proj_crop_dir = settings.crops_dir / project_id
    if not proj_crop_dir.exists():
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
    is_vietocr_fallback = False
    if (resolved_engine == "vietocr" or req.engine == "vietocr") and not is_vietocr_available():
        logger.warning(
            "CẢNH BÁO: Cấu hình engine=%s và language=%s nhưng vietocr chưa được cài đặt trong môi trường. Hệ thống chuyển sang RapidOCR.",
            req.engine,
            req.language,
        )
        resolved_engine = "rapidocr"
        is_vietocr_fallback = True

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

    if is_vietocr_fallback:
        await emit(
            WebSocketEvent(
                event="warning",
                project_id=req.project_id,
                job_id=job_id,
                progress=1.0,
                message=f"CẢNH BÁO: Cấu hình engine={req.engine} và language={req.language} nhưng vietocr chưa được cài đặt trong môi trường. Hệ thống chuyển sang RapidOCR.",
                extra={
                    "engine": resolved_engine,
                    "language": req.language,
                    "fallback_warning": True,
                },
            )
        )

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
        pts_table = await asyncio.to_thread(extract_video_pts_table, video_path)

        visual_filter = ThreeLayerVisualFilter(
            luma_threshold=req.luma_threshold,
            stability_frames=req.stability_frames,
            visual_diff_threshold=req.visual_diff_threshold,
            frame_step_sec=effective_step_sec,
            diagnostic_logging=req.diagnostic_logging,
            adaptive_threshold=req.adaptive_threshold,
            max_stroke_density=req.max_stroke_density,
        )
        deduplicator = FuzzySubtitleDeduplicator(
            fuzzy_match_threshold=req.fuzzy_match_threshold,
            max_merge_gap_sec=req.max_merge_gap_sec,
            mask_visual_diff_threshold=req.mask_visual_diff_threshold,
        )

        loop = asyncio.get_running_loop()
        span_queue: asyncio.Queue[tuple[VisualSubtitleSpan, float] | dict | None] = (
            asyncio.Queue()
        )

        def _frame_scanner_worker() -> None:
            """Decode video frames, crop ROI + Exclusion Masks, and run 3-Layer OpenCV Filter with Ring Buffer refinement."""
            try:
                cv2.setNumThreads(2)
            except Exception:
                pass
            try:
                stream = VideoROIStream(
                    video_path,
                    req.roi_config,
                    meta=meta,
                    pad_top_ratio=req.pad_top_ratio,
                    pad_bottom_ratio=req.pad_bottom_ratio,
                    pad_left_ratio=req.pad_left_ratio,
                    pad_right_ratio=req.pad_right_ratio,
                    pts_table=pts_table,
                )
            except Exception as err:
                loop.call_soon_threadsafe(
                    span_queue.put_nowait,
                    {
                        "type": "error",
                        "error": RuntimeError(f"Không thể mở video: {video_path}: {err}"),
                    },
                )
                loop.call_soon_threadsafe(span_queue.put_nowait, None)
                return

            try:
                last_progress_report = 0.0
                prev_coarse_idx: int | None = None
                prev_had_text: bool = False
                prev_coarse_mask: np.ndarray | None = None

                # In-memory sliding buffer of intermediate frames (f_idx, roi_bgr) between coarse steps
                # Completely eliminates cap.set() seeking overhead!
                window_buffer: list[tuple[int, np.ndarray]] = []
                total_frames_read: int = 0

                while not cancel_ev.is_set():
                    # Lightweight backpressure: avoid FFmpeg racing at 100% CPU far ahead of OCR
                    while not cancel_ev.is_set() and span_queue.qsize() >= 16:
                        time.sleep(0.05)
                    if cancel_ev.is_set():
                        break

                    frame_item = stream.read_frame()
                    if frame_item is None:
                        break
                    total_frames_read += 1
                    f_idx, timestamp_sec, roi_bgr = frame_item

                    # Buffer intermediate frames for instant zero-seek boundary refinement
                    if frame_step > 1 and f_idx % frame_step != 0:
                        window_buffer.append((f_idx, roi_bgr))
                        continue

                    curr_mask = extract_text_binary_mask(
                        roi_bgr,
                        luma_threshold=req.luma_threshold,
                        adaptive=req.adaptive_threshold,
                    )
                    curr_has_text = has_text_activity(
                        curr_mask, max_ratio=req.max_stroke_density
                    )
                    pct = min(95.0, 5.0 + (timestamp_sec / duration_sec) * 88.0)

                    refined_start_time: float | None = None
                    refined_end_time: float | None = None

                    # Zero-seek in-memory boundary refinement on state transitions
                    if frame_step > 1 and prev_coarse_idx is not None and window_buffer:
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
                            # 1. Text appearance: blank -> text (binary search on in-memory window_buffer)
                            if not prev_had_text and curr_has_text:
                                low, high = 0, len(window_buffer) - 1
                                ans_offset = len(window_buffer)
                                while low <= high:
                                    mid = (low + high) // 2
                                    m_mid = extract_text_binary_mask(
                                        window_buffer[mid][1],
                                        luma_threshold=req.luma_threshold,
                                        adaptive=req.adaptive_threshold,
                                    )
                                    if has_text_activity(
                                        m_mid, max_ratio=req.max_stroke_density
                                    ):
                                        ans_offset = mid
                                        high = mid - 1
                                    else:
                                        low = mid + 1
                                found_idx = (
                                    window_buffer[ans_offset][0]
                                    if ans_offset < len(window_buffer)
                                    else f_idx
                                )
                                refined_start_time = round(
                                    get_timestamp_for_frame(found_idx, pts_table, video_fps), 3
                                )

                            # 2. Text disappearance: text -> blank (binary search on in-memory window_buffer)
                            elif prev_had_text and not curr_has_text:
                                low, high = 0, len(window_buffer) - 1
                                ans_offset = -1
                                wb_masks: dict[int, np.ndarray] = {}
                                while low <= high:
                                    mid = (low + high) // 2
                                    m_mid = extract_text_binary_mask(
                                        window_buffer[mid][1],
                                        luma_threshold=req.luma_threshold,
                                        adaptive=req.adaptive_threshold,
                                    )
                                    wb_masks[mid] = m_mid
                                    if has_text_activity(
                                        m_mid, max_ratio=req.max_stroke_density
                                    ):
                                        ans_offset = mid
                                        low = mid + 1
                                    else:
                                        high = mid - 1
                                if (
                                    ans_offset >= 0
                                    and visual_filter._state == "STABILIZING"
                                    and visual_filter._pending_span is not None
                                    and visual_filter._pending_span.sampled_frames
                                    <= visual_filter.stability_frames
                                ):
                                    # Short subtitle was active across prev_coarse_idx and intermediate frames:
                                    # feed the intermediate text frame so STABILIZING has >= 2 frames before closing
                                    wb_idx, wb_roi = window_buffer[ans_offset]
                                    wb_t = round(
                                        get_timestamp_for_frame(wb_idx, pts_table, video_fps), 3
                                    )
                                    m_wb = wb_masks[ans_offset]
                                    for sp in visual_filter.process_frame(
                                        wb_roi, wb_t, precomputed_mask=m_wb
                                    ):
                                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, pct))
                                last_text_idx = (
                                    window_buffer[ans_offset][0]
                                    if ans_offset >= 0
                                    else prev_coarse_idx
                                )
                                refined_end_time = round(
                                    get_timestamp_for_frame(
                                        last_text_idx + 1, pts_table, video_fps
                                    ),
                                    3,
                                )

                            # 3. Direct cut between subtitle lines
                            elif is_cut:
                                assert prev_coarse_mask is not None
                                low, high = 0, len(window_buffer) - 1
                                ans_offset = len(window_buffer)
                                while low <= high:
                                    mid = (low + high) // 2
                                    m_mid = extract_text_binary_mask(
                                        window_buffer[mid][1],
                                        luma_threshold=req.luma_threshold,
                                        adaptive=req.adaptive_threshold,
                                    )
                                    if (
                                        not has_text_activity(
                                            m_mid, max_ratio=req.max_stroke_density
                                        )
                                        or compute_mask_visual_diff(
                                            prev_coarse_mask, m_mid
                                        )
                                        >= req.visual_diff_threshold
                                    ):
                                        ans_offset = mid
                                        high = mid - 1
                                    else:
                                        low = mid + 1
                                cut_idx = (
                                    window_buffer[ans_offset][0]
                                    if ans_offset < len(window_buffer)
                                    else f_idx
                                )
                                cut_t = round(
                                    get_timestamp_for_frame(cut_idx, pts_table, video_fps), 3
                                )
                                refined_end_time = cut_t
                                new_start_t = cut_t
                                if ans_offset < len(window_buffer):
                                    m_cut = extract_text_binary_mask(
                                        window_buffer[ans_offset][1],
                                        luma_threshold=req.luma_threshold,
                                        adaptive=req.adaptive_threshold,
                                    )
                                    if not has_text_activity(
                                        m_cut, max_ratio=req.max_stroke_density
                                    ):
                                        # There was a brief blank gap between old subtitle and new subtitle inside window_buffer
                                        low2, high2 = ans_offset + 1, len(window_buffer) - 1
                                        start_offset = len(window_buffer)
                                        while low2 <= high2:
                                            mid2 = (low2 + high2) // 2
                                            m_mid2 = extract_text_binary_mask(
                                                window_buffer[mid2][1],
                                                luma_threshold=req.luma_threshold,
                                                adaptive=req.adaptive_threshold,
                                            )
                                            if has_text_activity(
                                                m_mid2, max_ratio=req.max_stroke_density
                                            ):
                                                start_offset = mid2
                                                high2 = mid2 - 1
                                            else:
                                                low2 = mid2 + 1
                                        new_start_idx = (
                                            window_buffer[start_offset][0]
                                            if start_offset < len(window_buffer)
                                            else f_idx
                                        )
                                        new_start_t = round(
                                            get_timestamp_for_frame(
                                                new_start_idx, pts_table, video_fps
                                            ),
                                            3,
                                        )
                                refined_start_time = new_start_t

                        elif not prev_had_text and not curr_has_text:
                            # Detect short sub-second subtitles that appeared and disappeared within intermediate buffer
                            sub_found_start = None
                            sub_found_end = None
                            wb_mask_cache: dict[int, np.ndarray] = {}
                            for w_i, (wb_idx, wb_roi) in enumerate(window_buffer):
                                m_wb = extract_text_binary_mask(
                                    wb_roi,
                                    luma_threshold=req.luma_threshold,
                                    adaptive=req.adaptive_threshold,
                                )
                                wb_mask_cache[w_i] = m_wb
                                if has_text_activity(
                                    m_wb, max_ratio=req.max_stroke_density
                                ):
                                    if sub_found_start is None:
                                        sub_found_start = w_i
                                    sub_found_end = w_i
                            if sub_found_start is not None and sub_found_end is not None:
                                sub_end_idx = window_buffer[sub_found_end][0]
                                sub_end_t = round(
                                    get_timestamp_for_frame(
                                        sub_end_idx + 1, pts_table, video_fps
                                    ),
                                    3,
                                )
                                for w_i in range(sub_found_start, sub_found_end + 1):
                                    wb_idx, wb_roi = window_buffer[w_i]
                                    wb_t = round(
                                        get_timestamp_for_frame(wb_idx, pts_table, video_fps), 3
                                    )
                                    m_wb = wb_mask_cache[w_i]
                                    int_spans = visual_filter.process_frame(wb_roi, wb_t, precomputed_mask=m_wb)
                                    for sp in int_spans:
                                        sp.end_time = max(
                                            round(sp.start_time + 0.05, 3),
                                            min(sp.end_time, sub_end_t),
                                        )
                                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, pct))
                                for sp in visual_filter.flush():
                                    sp.end_time = max(
                                        round(sp.start_time + 0.05, 3),
                                        min(sp.end_time, sub_end_t),
                                    )
                                    loop.call_soon_threadsafe(span_queue.put_nowait, (sp, pct))

                    window_buffer.clear()

                    completed_spans = visual_filter.process_frame(
                        roi_bgr, timestamp_sec, precomputed_mask=curr_mask
                    )

                    if refined_start_time is not None:
                        if (
                            visual_filter._state == "ACTIVE"
                            and visual_filter._active_span is not None
                            and visual_filter._active_span.sampled_frames == 1
                        ):
                            visual_filter._active_span.start_time = refined_start_time
                        elif (
                            visual_filter._state == "STABILIZING"
                            and visual_filter._pending_span is not None
                            and visual_filter._pending_span.sampled_frames == 1
                        ):
                            visual_filter._pending_span.start_time = refined_start_time

                    if refined_end_time is not None:
                        for sp in completed_spans:
                            sp.end_time = max(round(sp.start_time + 0.05, 3), refined_end_time)

                    for sp in completed_spans:
                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, pct))

                    prev_coarse_idx = f_idx
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

                if not cancel_ev.is_set():
                    for sp in visual_filter.flush():
                        loop.call_soon_threadsafe(span_queue.put_nowait, (sp, 95.0))

                loop.call_soon_threadsafe(
                    span_queue.put_nowait,
                    {"type": "scanner_summary", "total_frames_read": total_frames_read},
                )
                loop.call_soon_threadsafe(span_queue.put_nowait, None)
            except Exception as err:
                loop.call_soon_threadsafe(
                    span_queue.put_nowait, {"type": "error", "error": err}
                )
                loop.call_soon_threadsafe(span_queue.put_nowait, None)
            finally:
                stream.close()

        scanner_task = asyncio.create_task(asyncio.to_thread(_frame_scanner_worker))
        raw_crop_counter = 1
        total_ocr_invocations = 0
        total_frames_read = 0
        filtered_empty_count = 0
        filtered_low_confidence_count = 0
        merged_duplicates_count = 0
        fragment_votes: dict[str, dict[str, dict[str, Any]]] = {}
        prev_was_timeout_split = False

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
                    elif item.get("type") == "scanner_summary":
                        total_frames_read = item.get("total_frames_read", 0)
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

                target_bgr: np.ndarray | None = None
                _safe_recognize: Any = None

                if resolved_engine == "gemini" and len(batch_spans) > 1:
                    total_ocr_invocations += len(batch_spans)
                    ocr_results = await asyncio.to_thread(
                        ocr_engine_cache.recognize_gemini_batch,
                        [sp.best_frame_bgr for sp, _ in batch_spans],
                        req.language,
                        req.gemini_api_key,
                    )
                else:
                    span_obj = batch_spans[0][0]
                    raw_bgr = span_obj.best_frame_bgr
                    # Downscale only excessively high-resolution crops (e.g. 4K/2K) to preserve contour detail
                    target_bgr = _downscale_crop_for_ocr(raw_bgr)

                    def _safe_recognize(img_target: np.ndarray, **ocr_kwargs: Any) -> OCRFrameResult:
                        b_thresh = ocr_kwargs.get("box_thresh", req.rapidocr_box_thresh)
                        u_ratio = ocr_kwargs.get("unclip_ratio", req.rapidocr_unclip_ratio)
                        t_score = ocr_kwargs.get("text_score", req.rapidocr_text_score)
                        thresh_val = ocr_kwargs.get("thresh", req.rapidocr_thresh)
                        try:
                            return ocr_engine_cache.recognize_frame(
                                img_target,
                                resolved_engine,
                                req.language,
                                req.gemini_api_key,
                                text_score=t_score,
                                box_thresh=b_thresh,
                                thresh=thresh_val,
                                unclip_ratio=u_ratio,
                                diagnostic_logging=req.diagnostic_logging,
                            )
                        except TypeError:
                            return ocr_engine_cache.recognize_frame(
                                img_target,
                                resolved_engine,
                                req.language,
                                req.gemini_api_key,
                            )

                    total_ocr_invocations += 1
                    single_res = await asyncio.to_thread(_safe_recognize, target_bgr)

                    # Multi-frame min-pooling + OCR voting with early-exit:
                    # - Skip extra OCR calls when the first pass is already clean and high-confidence (>= 0.94).
                    # - When voting runs, stop as soon as 2 candidates reach consensus on clean text (>= 0.85).
                    single_conf = float(single_res.confidence or 0.0)
                    single_clean = clean_ocr_text(single_res.text)
                    is_high_conf_clean = (
                        bool(single_clean)
                        and single_conf >= 0.94
                        and len(single_clean) >= 2
                        and _cjk_noise_penalty(single_clean) == 0
                    )
                    needs_voting = (
                        not single_clean
                        or single_conf < 0.80
                        or (is_cjk_string(single_clean) and not is_high_conf_clean)
                    )
                    if needs_voting and len(span_obj.candidate_frames) > 1:
                        voted_candidates: list[tuple[OCRFrameResult, np.ndarray]] = [(single_res, target_bgr)]
                        seen_keys: dict[str, int] = {}
                        if single_clean:
                            k0 = convert_t2s(single_clean).lower()
                            seen_keys[k0] = 1

                        def _has_consensus(res_obj: OCRFrameResult) -> bool:
                            txt_c = clean_ocr_text(res_obj.text)
                            if not txt_c or float(res_obj.confidence or 0.0) < 0.85:
                                return False
                            key = convert_t2s(txt_c).lower()
                            seen_keys[key] = seen_keys.get(key, 0) + 1
                            return seen_keys[key] >= 2

                        consensus_reached = False
                        pooled_raw = compute_min_pooled_frame(span_obj.candidate_frames)
                        if pooled_raw is not None:
                            pooled_target = _downscale_crop_for_ocr(pooled_raw)
                            total_ocr_invocations += 1
                            pooled_res = await asyncio.to_thread(_safe_recognize, pooled_target)
                            voted_candidates.append((pooled_res, pooled_target))
                            consensus_reached = _has_consensus(pooled_res)

                        if not consensus_reached:
                            for alt_cand in span_obj.candidate_frames[1:3]:
                                alt_target = _downscale_crop_for_ocr(alt_cand)
                                total_ocr_invocations += 1
                                res_alt = await asyncio.to_thread(_safe_recognize, alt_target)
                                voted_candidates.append((res_alt, alt_target))
                                if _has_consensus(res_alt):
                                    break

                        single_res, target_bgr = select_voted_ocr_result(voted_candidates)

                    ocr_results = [single_res]

                for (span, pct), ocr_res in zip(batch_spans, ocr_results):
                    if cancel_ev.is_set():
                        break

                    # Generic geometric mask-guided recovery for truncated boundary characters
                    if span.best_mask is not None and ocr_res.lines:
                        current_target_bgr = target_bgr if (resolved_engine != "gemini" and target_bgr is not None) else span.best_frame_bgr
                        if current_target_bgr is not None:
                            ocr_res = recover_truncated_text_lines(
                                image_bgr=current_target_bgr,
                                roi_mask=span.best_mask,
                                ocr_res=ocr_res,
                                recognize_fn=_safe_recognize if resolved_engine != "gemini" else None,
                                min_overflow_ratio=0.75,
                                diagnostic_logging=req.diagnostic_logging,
                                timestamp_sec=span.start_time,
                            )

                    cleaned_text = clean_cjk_subtitle_text(clean_ocr_text(ocr_res.text))
                    if cleaned_text and req.language in ("ch", "ja", "ko") and not is_cjk_string(cleaned_text):
                        # Drop pure non-CJK background hallucinations (e.g. "YA", "WA", "WW") in CJK mode
                        cleaned_text = ""
                    if not cleaned_text:
                        if ocr_res.lines:
                            filtered_low_confidence_count += 1
                            if req.diagnostic_logging:
                                logger.info(
                                    "[OCR DIAG] Frame at %.3fs: dropped subtitle segment due to min_confidence (confidence=%.3f)",
                                    span.start_time,
                                    ocr_res.confidence,
                                )
                        else:
                            filtered_empty_count += 1
                            if req.diagnostic_logging:
                                logger.info(
                                    "[OCR DIAG] Frame at %.3fs: dropped subtitle segment due to text_score (no text detected or text_score below threshold)",
                                    span.start_time,
                                )
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

                    norm_text = convert_t2s(cleaned_text)
                    trad_cnt, simp_cnt = count_trad_simp_characters(cleaned_text)
                    is_mixed = is_mixed_cjk_script(cleaned_text)
                    is_low_conf = is_mixed or (ocr_res.confidence < 0.60)

                    candidate_seg = SubtitleSegment(
                        id=str(uuid.uuid4()),
                        project_id=req.project_id,
                        index=len(deduplicator.segments) + 1,
                        start_time=round(span.start_time, 3),
                        end_time=round(max(span.start_time + 0.05, span.end_time), 3),
                        text=cleaned_text,
                        original_text=cleaned_text,
                        normalized_text=norm_text,
                        confidence=round(float(ocr_res.confidence or 0.85), 4),
                        is_low_confidence=is_low_conf,
                        source="ocr",
                        crop_image_url=crop_url,
                        words=[],
                    )

                    dedup_mask = span.best_mask
                    orig_max_gap = deduplicator.max_merge_gap_sec
                    if deduplicator.segments:
                        prev_seg = deduplicator.segments[-1]
                        seg_gap = max(0.0, candidate_seg.start_time - prev_seg.end_time)
                        sim_score = compute_ocr_text_similarity(prev_seg.text, candidate_seg.text)
                        # Bridge adjacent background-split or 0.65s-timeout-split fragments across all languages:
                        # when text matches the previous line, extend prev_seg instead of creating a new line.
                        bridge_gap = 0.25 if prev_was_timeout_split else 0.15
                        is_containment_split = (
                            seg_gap <= bridge_gap
                            and (
                                is_same_cjk_subtitle_fragment(prev_seg.text, candidate_seg.text)
                                or (
                                    sim_score < max(85.0, req.fuzzy_match_threshold)
                                    and is_same_subtitle_fragment(prev_seg.text, candidate_seg.text)
                                )
                            )
                        )
                        if is_containment_split:
                            # Duration-weighted vote across all fragments of this subtitle line:
                            # the reading that stays on screen longest wins (noise readings are brief).
                            votes = fragment_votes.setdefault(prev_seg.id or "", {})
                            if not votes:
                                add_fragment_vote(
                                    votes,
                                    prev_seg.text,
                                    prev_seg.end_time - prev_seg.start_time,
                                    float(prev_seg.confidence),
                                    prev_seg.original_text,
                                    prev_seg.normalized_text,
                                    prev_seg.crop_image_url,
                                )
                            add_fragment_vote(
                                votes,
                                candidate_seg.text,
                                candidate_seg.end_time - candidate_seg.start_time,
                                float(candidate_seg.confidence),
                                candidate_seg.original_text,
                                candidate_seg.normalized_text,
                                candidate_seg.crop_image_url,
                            )
                            prev_seg.end_time = round(max(prev_seg.end_time, candidate_seg.end_time), 3)
                            winner = select_fragment_vote(votes)
                            prev_seg.text = winner["text"]
                            prev_seg.original_text = winner["original_text"] or winner["text"]
                            prev_seg.normalized_text = winner["normalized_text"] or convert_t2s(winner["text"])
                            prev_seg.confidence = round(winner["confidence"], 4)
                            if winner["crop_image_url"]:
                                prev_seg.crop_image_url = winner["crop_image_url"]
                            if req.diagnostic_logging:
                                logger.info(
                                    "[OCR DIAG] Bridged fragment at %.3fs into [%.3fs -> %.3fs] '%s'",
                                    candidate_seg.start_time,
                                    prev_seg.start_time,
                                    prev_seg.end_time,
                                    prev_seg.text,
                                )
                            merged_duplicates_count += 1
                            await db.update_project_status(
                                req.project_id, status=JobStatus.RUNNING, progress=pct
                            )
                            if prev_seg.id:
                                await db.update_segment(
                                    req.project_id,
                                    prev_seg.id,
                                    SubtitleSegmentUpdate(
                                        start_time=prev_seg.start_time,
                                        end_time=prev_seg.end_time,
                                        text=prev_seg.text,
                                        original_text=prev_seg.original_text,
                                        normalized_text=prev_seg.normalized_text,
                                        confidence=prev_seg.confidence,
                                        is_low_confidence=prev_seg.is_low_confidence,
                                        crop_image_url=prev_seg.crop_image_url,
                                    ),
                                )
                                await emit(
                                    WebSocketEvent(
                                        event="segment_updated",
                                        project_id=req.project_id,
                                        job_id=job_id,
                                        progress=round(pct, 1),
                                        message=f"Gộp trùng lặp [{prev_seg.start_time:.1f}s -> {prev_seg.end_time:.1f}s] {prev_seg.text.replace(chr(10), ' ')}",
                                        segment=prev_seg,
                                    )
                                )
                            prev_was_timeout_split = span.is_timeout_split
                            continue
                        if (
                            seg_gap <= max(orig_max_gap, 0.65)
                            and sim_score >= max(85.0, req.fuzzy_match_threshold)
                        ):
                            dedup_mask = None
                            deduplicator.max_merge_gap_sec = max(orig_max_gap, 0.65)
                    try:
                        active_seg, was_merged = deduplicator.add_or_merge(
                            candidate_seg, mask=dedup_mask
                        )
                    finally:
                        deduplicator.max_merge_gap_sec = orig_max_gap
                    prev_was_timeout_split = span.is_timeout_split
                    await db.update_project_status(
                        req.project_id, status=JobStatus.RUNNING, progress=pct
                    )

                    if was_merged and active_seg.id:
                        merged_duplicates_count += 1
                        if req.diagnostic_logging:
                            logger.info(
                                "[OCR DIAG] Segment at %.3fs: dropped due to deduplication merge into [%.3fs -> %.3fs] '%s'",
                                candidate_seg.start_time,
                                active_seg.start_time,
                                active_seg.end_time,
                                active_seg.text,
                            )
                        await db.update_segment(
                            req.project_id,
                            active_seg.id,
                            SubtitleSegmentUpdate(
                                start_time=active_seg.start_time,
                                end_time=active_seg.end_time,
                                text=active_seg.text,
                                original_text=active_seg.original_text,
                                normalized_text=active_seg.normalized_text,
                                confidence=active_seg.confidence,
                                is_low_confidence=active_seg.is_low_confidence,
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

        # Post-pass CJK majority noise filter (works in both explicit CJK and 'auto' language modes)
        kept_segs, dropped_segs = filter_cjk_majority_noise_segments(
            deduplicator.segments, req.language
        )
        if dropped_segs:
            for dseg in dropped_segs:
                if dseg.id:
                    await db.delete_segment(req.project_id, dseg.id)
            deduplicator.segments = kept_segs
            filtered_low_confidence_count += len(dropped_segs)

        elapsed = round(time.perf_counter() - start_wall, 2)
        elapsed_ms = round((time.perf_counter() - start_wall) * 1000.0, 1)
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

        diag_stats = {
            "total_frames_read": total_frames_read,
            "total_frames_sampled": visual_filter.total_frames_sampled,
            "frames_with_text": visual_filter.frames_with_text,
            "ocr_invocations": total_ocr_invocations,
            "filtered_low_confidence": filtered_low_confidence_count,
            "filtered_empty_or_short": filtered_empty_count,
            "merged_duplicates": merged_duplicates_count,
            "retained_segments": len(deduplicator.segments),
            "elapsed_ms": elapsed_ms,
            "elapsed_sec": elapsed,
        }
        if req.diagnostic_logging:
            logger.info(
                "[OCR DIAGNOSTICS] Project %s summary: Total frames read: %d, OCR calls: %d, Total processing time: %.1fms (%.2fs), Stats: %s",
                req.project_id,
                total_frames_read,
                total_ocr_invocations,
                elapsed_ms,
                elapsed,
                diag_stats,
            )

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
                    "diagnostic_stats": diag_stats,
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
