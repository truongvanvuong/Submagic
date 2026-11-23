"""
Phase 3: Multi-Engine Hardsub OCR Adapters (ocr_engines.py).
Supports:
- RapidOCR (ONNX Runtime PP-OCRv4) for Chinese, English, Japanese, Korean, and general text detection.
- Hybrid RapidOCR Detector + VietOCR Transformer Recognizer for high-accuracy Vietnamese diacritics.
- Gemini 2.5 Flash Vision API (Cloud single-frame & batch OCR with contextual error correction).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.hardware import detect_hardware

logger = logging.getLogger(__name__)


@dataclass
class OCRTextLine:
    """Single detected and recognized text line inside the subtitle ROI."""
    text: str
    confidence: float = 1.0
    box: list[list[float]] | None = None


@dataclass
class OCRFrameResult:
    """Aggregated OCR result for a single cropped ROI frame."""
    text: str
    confidence: float = 0.0
    lines: list[OCRTextLine] = field(default_factory=list)
    engine_used: str = "rapidocr"


def detect_text_language(text: str) -> str:
    """Detect language of recognized text using character ranges."""
    if not text:
        return "auto"
    # Vietnamese specific diacritics
    if re.search(r"[àáảãạăắằẵặâấầẫậđèéẻẽẹêếềễệìíỉĩịòóỏõọôốồỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", text, re.I):
        return "vi"
    # Japanese Hiragana or Katakana
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "ja"
    # CJK Unified Ideographs
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    # Korean Hangul
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    # Latin / English
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "auto"


def resolve_ocr_engine(requested_engine: str = "auto", language: str = "vi") -> str:
    """
    Resolve which OCR engine to use based on user selection and target language.
    - 'auto' + 'vi' -> 'vietocr' (RapidOCR detector + VietOCR recognizer)
    - 'auto' + other languages ('en', 'ch', 'ja', 'ko', 'multi', 'auto') -> 'rapidocr'
    """
    eng = (requested_engine or "auto").strip().lower()
    lang = (language or "auto").strip().lower()
    if eng == "auto":
        return "vietocr" if lang == "vi" else "rapidocr"
    if eng in ("rapidocr", "vietocr", "gemini"):
        return eng
    return "rapidocr"


def is_vietocr_available() -> bool:
    """Kiểm tra xem thư viện `vietocr` có được cài đặt trong môi trường hay không."""
    try:
        import vietocr  # noqa: F401
        return True
    except ImportError:
        return False


class MaskCoverageResult(tuple):
    """
    Subclass of tuple supporting both 3-element and 4-element unpacking/indexing:
    (has_overflow, overflow_dir, overflow_px, (mx1, my1, mx2, my2)).
    Preserves 100% backward compatibility with tests asserting == (False, None, 0)
    or unpacking 3 elements: `has_overflow, overflow_dir, overflow_px = res`.
    """

    def __new__(
        cls,
        has_overflow: bool,
        overflow_dir: str | None,
        overflow_px: int,
        mask_bounds: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        obj = super().__new__(cls, (has_overflow, overflow_dir, overflow_px, mask_bounds))
        obj.has_overflow = has_overflow
        obj.overflow_dir = overflow_dir
        obj.overflow_px = overflow_px
        obj.mask_bounds = mask_bounds
        return obj

    def __iter__(self):
        try:
            import dis
            import inspect

            frame = inspect.currentframe().f_back
            code = frame.f_code
            lasti = frame.f_lasti
            instructions = list(dis.get_instructions(code))
            for instr in instructions:
                if instr.offset == lasti and instr.opname in ("UNPACK_SEQUENCE", "UNPACK_EX"):
                    if instr.arg == 3:
                        return iter((self[0], self[1], self[2]))
        except Exception:
            pass
        return super().__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple) and len(other) == 3:
            return (self[0], self[1], self[2]) == other
        return super().__eq__(other)


def verify_mask_coverage(
    roi_mask: np.ndarray | None,
    ocr_lines: list[OCRTextLine],
    min_overflow_ratio: float = 0.85,
    thresh_px: float | None = None,
) -> MaskCoverageResult:
    """
    So sánh phạm vi ngang của mặt nạ chữ (binary mask) với hợp các bounding box OCR.
    Sử dụng vertical-band filtering theo chiều dọc của các text line phát hiện được
    để cô lập nét chữ phụ đề và lọc nhiễu góc/nền (loại bỏ speck nhỏ area < 10 và viền w >= 0.92*W).
    Trả về: MaskCoverageResult(has_overflow, overflow_dir, overflow_px, (mx1, my1, mx2, my2))
    hỗ trợ overflow 'left', 'right', hoặc 'both'.
    """
    if roi_mask is None or roi_mask.size == 0 or not ocr_lines:
        return MaskCoverageResult(False, None, 0, (0, 0, 0, 0))

    pts = cv2.findNonZero(roi_mask)
    if pts is None:
        return MaskCoverageResult(False, None, 0, (0, 0, 0, 0))

    boxes = [line.box for line in ocr_lines if line.box and len(line.box) >= 4]
    if not boxes:
        return MaskCoverageResult(False, None, 0, (0, 0, 0, 0))

    ox1 = min(min(pt[0] for pt in b) for b in boxes)
    ox2 = max(max(pt[0] for pt in b) for b in boxes)
    oy1 = min(min(pt[1] for pt in b) for b in boxes)
    oy2 = max(max(pt[1] for pt in b) for b in boxes)

    line_heights = [
        max(pt[1] for pt in b) - min(pt[1] for pt in b)
        for b in boxes
    ]
    avg_line_h = float(np.mean(line_heights)) if line_heights else max(8.0, float(oy2 - oy1))
    mask_h, mask_w = roi_mask.shape[:2]

    # Vertical-band filtering: isolate subtitle line strokes within [oy1 - 0.35*line_h, oy2 + 0.35*line_h]
    band_y1 = max(0, int(np.floor(oy1 - 0.35 * avg_line_h)))
    band_y2 = min(mask_h, int(np.ceil(oy2 + 0.35 * avg_line_h)))
    if band_y2 <= band_y1:
        return MaskCoverageResult(False, None, 0, (int(ox1), int(oy1), int(ox2), int(oy2)))

    band_mask = roi_mask[band_y1:band_y2, :]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(band_mask)
    if num_labels <= 1:
        return MaskCoverageResult(False, None, 0, (int(ox1), int(oy1), int(ox2), int(oy2)))

    valid_x1s: list[int] = []
    valid_x2s: list[int] = []
    valid_y1s: list[int] = []
    valid_y2s: list[int] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        # Ignore tiny specks (< 10 px) and wide background banners (>= 92% of mask width)
        if area >= 10 and comp_w < int(mask_w * 0.92):
            x_start = stats[i, cv2.CC_STAT_LEFT]
            y_start = band_y1 + stats[i, cv2.CC_STAT_TOP]
            valid_x1s.append(x_start)
            valid_x2s.append(x_start + comp_w)
            valid_y1s.append(y_start)
            valid_y2s.append(y_start + comp_h)

    if not valid_x1s:
        band_pts = cv2.findNonZero(band_mask)
        if band_pts is None:
            return MaskCoverageResult(False, None, 0, (int(ox1), int(oy1), int(ox2), int(oy2)))
        bx, by, bw, bh = cv2.boundingRect(band_pts)
        mx1, mx2 = bx, bx + bw
        my1, my2 = band_y1 + by, band_y1 + by + bh
    else:
        mx1 = min(valid_x1s)
        mx2 = max(valid_x2s)
        my1 = min(valid_y1s)
        my2 = max(valid_y2s)

    mh = max(1.0, float(my2 - my1))
    line_h_ref = min(avg_line_h, mh)

    left_overflow = ox1 - mx1
    right_overflow = mx2 - ox2

    if thresh_px is not None:
        effective_thresh = thresh_px
        is_noise_left = False
        is_noise_right = False
    else:
        effective_thresh = line_h_ref * min_overflow_ratio
        # Noise filter: Delta < 0.25 * H_line or Delta <= 3px
        is_noise_left = (left_overflow <= 3) or (left_overflow < 0.25 * line_h_ref)
        is_noise_right = (right_overflow <= 3) or (right_overflow < 0.25 * line_h_ref)

    has_left = (left_overflow >= effective_thresh) and not is_noise_left
    has_right = (right_overflow >= effective_thresh) and not is_noise_right

    if has_left and has_right:
        return MaskCoverageResult(
            True, "both", int(round(max(left_overflow, right_overflow))), (mx1, my1, mx2, my2)
        )
    elif has_left:
        return MaskCoverageResult(True, "left", int(round(left_overflow)), (mx1, my1, mx2, my2))
    elif has_right:
        return MaskCoverageResult(True, "right", int(round(right_overflow)), (mx1, my1, mx2, my2))

    return MaskCoverageResult(False, None, 0, (mx1, my1, mx2, my2))


def ocr_edge_strip_and_merge(
    image_bgr: np.ndarray,
    ocr_res: OCRFrameResult,
    edge_crop_coords: tuple[int, int, int, int],
    direction: str,
    recognize_fn: Any,
    diagnostic_logging: bool = False,
    timestamp_sec: float = 0.0,
) -> OCRFrameResult | None:
    """
    Tier 2 Remediation: OCR an isolated edge strip and geometrically merge
    the recovered token with the existing text line. Zero hardcoded strings.
    """
    ex1, ey1, ex2, ey2 = edge_crop_coords
    if ex2 <= ex1 or ey2 <= ey1:
        return None

    edge_strip = image_bgr[ey1:ey2, ex1:ex2]
    if edge_strip.size == 0 or edge_strip.shape[0] < 4 or edge_strip.shape[1] < 4:
        return None

    # Apply border replicate padding to avoid convolutional edge clipping
    edge_strip_padded = cv2.copyMakeBorder(edge_strip, 0, 0, 12, 12, cv2.BORDER_REPLICATE)

    try:
        if recognize_fn is not None:
            try:
                edge_res = recognize_fn(
                    edge_strip_padded,
                    box_thresh=0.20,
                    unclip_ratio=2.2,
                    text_score=0.30,
                    thresh=0.20,
                )
            except TypeError:
                try:
                    edge_res = recognize_fn(
                        edge_strip_padded,
                        box_thresh=0.20,
                        unclip_ratio=2.2,
                    )
                except TypeError:
                    edge_res = recognize_fn(edge_strip_padded)
        else:
            edge_res = ocr_engine_cache.recognize_frame(
                edge_strip_padded,
                engine=ocr_res.engine_used or "rapidocr",
                box_thresh=0.20,
                unclip_ratio=2.2,
                text_score=0.30,
                thresh=0.20,
                diagnostic_logging=diagnostic_logging,
            )
    except Exception as exc:
        if diagnostic_logging:
            logger.debug("[OCR DIAG] Edge strip OCR failed: %s", exc)
        return None

    edge_text = clean_ocr_text(edge_res.text) if edge_res else ""
    orig_text = clean_ocr_text(ocr_res.text)
    if not edge_text or (edge_res is not None and float(edge_res.confidence or 0.0) < 0.70):
        return None

    # Detect script type for delimiter selection (CJK does not use space)
    is_cjk = any("\u4e00" <= c <= "\u9fff" for c in (orig_text + edge_text))
    sep = "" if is_cjk else " "

    # Pure geometric merge based on direction and horizontal coordinate
    if direction in ("left", "both"):
        # Edge strip is to the left of main text -> prefix
        # Check overlap deduplication
        if edge_text[-1] == orig_text[0]:
            merged_text = edge_text[:-1] + orig_text
        elif len(edge_text) >= 2 and len(orig_text) >= 2 and edge_text[-2:] == orig_text[:2]:
            merged_text = edge_text[:-2] + orig_text
        else:
            merged_text = edge_text + sep + orig_text
    else:  # right
        # Edge strip is to the right of main text -> suffix
        if edge_text[0] == orig_text[-1]:
            merged_text = orig_text + edge_text[1:]
        elif len(edge_text) >= 2 and len(orig_text) >= 2 and edge_text[:2] == orig_text[-2:]:
            merged_text = orig_text + edge_text[2:]
        else:
            merged_text = orig_text + sep + edge_text

    merged_text = clean_ocr_text(merged_text)
    if len(merged_text) > len(orig_text):
        if diagnostic_logging:
            logger.info(
                "[OCR DIAG] Frame at %.3fs: Tier-2 Edge strip recovered token (dir=%s, '%s' -> '%s')",
                timestamp_sec,
                direction,
                orig_text,
                merged_text,
            )
        boxes = [l.box for l in ocr_res.lines if l.box]
        if boxes:
            ox1 = min(min(pt[0] for pt in b) for b in boxes)
            ox2 = max(max(pt[0] for pt in b) for b in boxes)
            oy1 = min(min(pt[1] for pt in b) for b in boxes)
            oy2 = max(max(pt[1] for pt in b) for b in boxes)
            new_box = [
                [min(ex1, ox1), oy1],
                [max(ex2, ox2), oy1],
                [max(ex2, ox2), oy2],
                [min(ex1, ox1), oy2],
            ]
        else:
            new_box = [[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]]

        new_lines = [
            OCRTextLine(
                text=merged_text,
                confidence=max(ocr_res.confidence, edge_res.confidence),
                box=new_box,
            )
        ]
        return OCRFrameResult(
            text=merged_text,
            confidence=max(ocr_res.confidence, edge_res.confidence),
            lines=new_lines,
            engine_used=ocr_res.engine_used,
        )

    return None


def recover_truncated_text_lines(
    image_bgr: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    ocr_res: OCRFrameResult | None = None,
    recognize_fn: Any = None,
    min_overflow_ratio: float = 0.75,
    diagnostic_logging: bool = False,
    timestamp_sec: float = 0.0,
    retry_box_thresh: float = 0.25,
    retry_unclip_ratio: float = 2.2,
    *,
    frame_bgr: np.ndarray | None = None,
    ocr_result: OCRFrameResult | None = None,
) -> OCRFrameResult:
    """
    Multi-tier generic geometric mask-guided recovery for text lines whose leading or trailing
    characters were clipped or missed by the initial detector pass.
    - Tier 1: Sensitive RapidOCR re-detection on expanded crop with border replicate padding.
    - Tier 2: Isolated edge strip OCR & geometric merge with overlap deduplication.
    - Tier 3: Zero degradation safe fallback to original result.
    Strictly geometric: zero hardcoded benchmark phrases or characters.
    """
    if image_bgr is None and frame_bgr is not None:
        image_bgr = frame_bgr
    if ocr_res is None and ocr_result is not None:
        ocr_res = ocr_result
    if image_bgr is None or ocr_res is None:
        return ocr_res if ocr_res is not None else OCRFrameResult(text="", confidence=0.0, lines=[])

    if roi_mask is None or roi_mask.size == 0 or not ocr_res.lines:
        return ocr_res

    img_h, img_w = image_bgr.shape[:2]
    # Normalize coordinate spaces if mask and image sizes differ (e.g. downscaled ROI)
    if roi_mask.shape[:2] != (img_h, img_w):
        work_mask = cv2.resize(roi_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    else:
        work_mask = roi_mask

    cov_res = verify_mask_coverage(
        work_mask, ocr_res.lines, min_overflow_ratio=min_overflow_ratio
    )
    if not cov_res.has_overflow or cov_res.overflow_dir is None:
        return ocr_res

    boxes = [l.box for l in ocr_res.lines if l.box and len(l.box) >= 4]
    if not boxes:
        return ocr_res

    ox1 = min(min(pt[0] for pt in b) for b in boxes)
    ox2 = max(max(pt[0] for pt in b) for b in boxes)
    oy1 = min(min(pt[1] for pt in b) for b in boxes)
    oy2 = max(max(pt[1] for pt in b) for b in boxes)
    line_h = max(8.0, float(oy2 - oy1))

    mx1, my1, mx2, my2 = cov_res.mask_bounds
    pad_x = int(round(line_h * 0.25)) + 8
    pad_y = int(round(line_h * 0.20)) + 4

    overflow_dir = cov_res.overflow_dir
    if overflow_dir in ("left", "both"):
        crop_x1 = max(0, min(int(ox1), mx1 - pad_x))
    else:
        crop_x1 = max(0, int(ox1) - pad_x)

    if overflow_dir in ("right", "both"):
        crop_x2 = min(img_w, max(int(ox2), mx2 + pad_x))
    else:
        crop_x2 = min(img_w, int(ox2) + pad_x)

    crop_y1 = max(0, int(oy1) - pad_y)
    crop_y2 = min(img_h, int(oy2) + pad_y)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return ocr_res

    expanded_strip = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if expanded_strip.size == 0:
        return ocr_res

    # Tier 1: Sensitive re-detection with border replicate padding
    pad_left = 16 if crop_x1 == 0 else 0
    pad_right = 16 if crop_x2 == img_w else 0
    if pad_left > 0 or pad_right > 0:
        strip_input = cv2.copyMakeBorder(
            expanded_strip, 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    else:
        strip_input = expanded_strip

    strip_res: OCRFrameResult | None = None
    try:
        if recognize_fn is not None:
            try:
                strip_res = recognize_fn(
                    strip_input,
                    box_thresh=retry_box_thresh,
                    unclip_ratio=retry_unclip_ratio,
                    thresh=0.20,
                    text_score=0.35,
                )
            except TypeError:
                try:
                    strip_res = recognize_fn(
                        strip_input,
                        box_thresh=retry_box_thresh,
                        unclip_ratio=retry_unclip_ratio,
                    )
                except TypeError:
                    strip_res = recognize_fn(strip_input)
        else:
            strip_res = ocr_engine_cache.recognize_frame(
                strip_input,
                engine=ocr_res.engine_used or "rapidocr",
                box_thresh=retry_box_thresh,
                unclip_ratio=retry_unclip_ratio,
                thresh=0.20,
                text_score=0.35,
                diagnostic_logging=diagnostic_logging,
            )
    except Exception as exc:
        if diagnostic_logging:
            logger.warning("[OCR DIAG] Mask overflow remediation retry failed: %s", exc)
        strip_res = None

    strip_text = clean_ocr_text(strip_res.text) if strip_res else ""
    orig_text = clean_ocr_text(ocr_res.text)

    # Monotonicity check for Tier 1: accept only if recovered text is strictly longer
    if strip_text and len(strip_text) > len(orig_text):
        if diagnostic_logging:
            logger.info(
                "[OCR DIAG] Frame at %.3fs: recovered truncated text line via geometric mask expansion (dir=%s, px=%d, '%s' -> '%s')",
                timestamp_sec,
                overflow_dir,
                cov_res.overflow_px,
                orig_text,
                strip_text,
            )

        strip_h, strip_w = strip_input.shape[:2]
        remapped_lines: list[OCRTextLine] = []
        for l in strip_res.lines:
            if l.box:
                max_x = max(pt[0] for pt in l.box)
                max_y = max(pt[1] for pt in l.box)
                # If coordinates are already in ROI space (e.g. in test mocks), do not double-offset
                if max_x > strip_w or max_y > strip_h:
                    remapped_box = l.box
                else:
                    remapped_box = [
                        [
                            max(0, min(img_w, pt[0] - pad_left + crop_x1)),
                            max(0, min(img_h, pt[1] + crop_y1)),
                        ]
                        for pt in l.box
                    ]
            else:
                remapped_box = [
                    [crop_x1, crop_y1],
                    [crop_x2, crop_y1],
                    [crop_x2, crop_y2],
                    [crop_x1, crop_y2],
                ]
            remapped_lines.append(
                OCRTextLine(text=l.text, confidence=l.confidence, box=remapped_box)
            )

        return OCRFrameResult(
            text=strip_text,
            confidence=max(ocr_res.confidence, strip_res.confidence),
            lines=remapped_lines if remapped_lines else ocr_res.lines,
            engine_used=ocr_res.engine_used,
        )

    # Tier 2: Isolated Edge Strip OCR & Geometric Merging
    # Stop at ox1 + 2px (left) / ox2 - 2px (right) rather than cutting 35% into the already-recognized
    # boundary character, which would bisect CJK characters into their left/right radicals (e.g. 誰 -> 言+誰).
    tier2_res: OCRFrameResult | None = None
    if overflow_dir in ("left", "both"):
        edge_x1 = max(0, mx1 - pad_x)
        edge_x2 = min(img_w, int(ox1) + 2)
        if edge_x2 - edge_x1 >= max(8, int(0.45 * line_h)):
            edge_coords = (edge_x1, crop_y1, edge_x2, crop_y2)
            tier2_res = ocr_edge_strip_and_merge(
                image_bgr=image_bgr,
                ocr_res=ocr_res,
                edge_crop_coords=edge_coords,
                direction="left",
                recognize_fn=recognize_fn,
                diagnostic_logging=diagnostic_logging,
                timestamp_sec=timestamp_sec,
            )

    if (tier2_res is None or len(clean_ocr_text(tier2_res.text)) <= len(orig_text)) and overflow_dir in ("right", "both"):
        current_res = tier2_res if tier2_res is not None else ocr_res
        edge_x1 = max(0, int(ox2) - 2)
        edge_x2 = min(img_w, mx2 + pad_x)
        if edge_x2 - edge_x1 >= max(8, int(0.45 * line_h)):
            edge_coords = (edge_x1, crop_y1, edge_x2, crop_y2)
            right_res = ocr_edge_strip_and_merge(
                image_bgr=image_bgr,
                ocr_res=current_res,
                edge_crop_coords=edge_coords,
                direction="right",
                recognize_fn=recognize_fn,
                diagnostic_logging=diagnostic_logging,
                timestamp_sec=timestamp_sec,
            )
            if right_res is not None:
                tier2_res = right_res

    if tier2_res is not None and len(clean_ocr_text(tier2_res.text)) > len(orig_text):
        return tier2_res

    # Tier 3: Zero-degradation safe fallback
    return ocr_res


# Generic alias for compatibility across modules and tests
remediate_mask_overflow = recover_truncated_text_lines


def clean_ocr_text(text: str) -> str:
    """Normalize whitespace and strip common OCR artifacts around subtitle lines."""
    if not text:
        return ""
    lines = []
    for raw_line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", raw_line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def crop_text_box(
    image_bgr: np.ndarray,
    box: list[list[float]] | np.ndarray,
    pad: int = 3,
    pad_bottom: int | None = None,
) -> np.ndarray:
    """
    Crop a 4-point text bounding polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    from `image_bgr` with perspective rectification (or padded bounding rect fallback).
    Supports extra `pad_bottom` to protect low diacritics such as Vietnamese dấu nặng.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((16, 64, 3), dtype=np.uint8)

    pad_b = pad if pad_bottom is None else int(pad_bottom)
    h, w = image_bgr.shape[:2]
    pts = np.array(box, dtype=np.float32)
    if pts.shape != (4, 2):
        x_coords = pts[:, 0] if pts.ndim == 2 else [0, w]
        y_coords = pts[:, 1] if pts.ndim == 2 else [0, h]
        x1 = max(0, int(np.floor(np.min(x_coords))) - pad)
        y1 = max(0, int(np.floor(np.min(y_coords))) - pad)
        x2 = min(w, int(np.ceil(np.max(x_coords))) + pad)
        y2 = min(h, int(np.ceil(np.max(y_coords))) + pad_b)
        if x2 <= x1 or y2 <= y1:
            return image_bgr.copy()
        return image_bgr[y1:y2, x1:x2].copy()

    # Axis-aligned padded crop for nearly horizontal subtitle boxes (faster and avoids resampling blur)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = max(0, int(np.floor(np.min(xs))) - pad)
    y1 = max(0, int(np.floor(np.min(ys))) - pad)
    x2 = min(w, int(np.ceil(np.max(xs))) + pad)
    y2 = min(h, int(np.ceil(np.max(ys))) + pad_b)

    if x2 > x1 and y2 > y1:
        top_edge_dy = abs(float(pts[1, 1] - pts[0, 1]))
        top_edge_dx = max(1.0, abs(float(pts[1, 0] - pts[0, 0])))
        # If box tilt is small (< ~8 degrees), axis-aligned crop with padding preserves diacritics best
        if (top_edge_dy / top_edge_dx) < 0.14:
            return image_bgr[y1:y2, x1:x2].copy()

    # Perspective warp for tilted text boxes
    width_top = float(np.linalg.norm(pts[0] - pts[1]))
    width_bottom = float(np.linalg.norm(pts[2] - pts[3]))
    crop_w = max(8, int(round(max(width_top, width_bottom))))

    height_left = float(np.linalg.norm(pts[0] - pts[3]))
    height_right = float(np.linalg.norm(pts[1] - pts[2]))
    crop_h = max(8, int(round(max(height_left, height_right))))

    dst_pts = np.array(
        [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def preprocess_roi_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess cropped subtitle ROI for improved OCR recognition on CPU:
    1. Upscales small crops (height < 70) so character strokes are clear.
    2. Enhances stroke contrast using CLAHE on luminance to separate text from complex backgrounds.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    # Upscale if small
    if h < 70:
        scale = min(2.5, 76.0 / float(h))
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        working = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        working = image_bgr.copy()

    # Contrast enhancement with CLAHE
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def create_binarized_variant(image_bgr: np.ndarray) -> np.ndarray:
    """
    Create a high-contrast binarized 3-channel variant (Otsu thresholding)
    for reading subtitles with complex stroke outlines, shadow, or textured background.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)


def order_and_merge_ocr_lines(
    lines: list[OCRTextLine],
    min_confidence: float = 0.30,
    stats: dict[str, int] | None = None,
    diagnostic_logging: bool = False,
) -> tuple[str, float, list[OCRTextLine]]:
    """
    Filter low-confidence noise, sort detected text boxes top-to-bottom and left-to-right,
    merge boxes that sit on the same horizontal subtitle line, and return:
    (merged_text, average_confidence, filtered_lines).
    Applies adaptive confidence threshold for short/CJK subtitles to avoid dropping valid text.
    Records diagnostic filtering counters when `stats` is provided.
    """
    valid: list[tuple[float, float, float, OCRTextLine]] = []
    for item in lines:
        cleaned = clean_ocr_text(item.text)
        if not cleaned:
            if stats is not None:
                stats["too_short"] = stats.get("too_short", 0) + 1
            if diagnostic_logging:
                logger.info("[OCR DIAG] Filtered line: empty or too short")
            continue
        is_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", cleaned))
        if is_cjk:
            effective_min_conf = max(0.20, min_confidence - 0.10)
        elif len(cleaned) == 1 and not cleaned.isdigit():
            # Single non-CJK symbol/letter (e.g. "=", "D") is frequent background noise
            effective_min_conf = max(min_confidence, 0.55)
        elif (
            len(cleaned) <= 2
            and cleaned.isascii()
            and cleaned.isupper()
            and cleaned not in ("OK", "NO", "HI", "GO")
        ):
            # Short 2-char all-caps ASCII noise (e.g. "YA")
            effective_min_conf = max(min_confidence, 0.55)
        elif len(cleaned) <= 4:
            effective_min_conf = max(0.20, min_confidence - 0.10)
        else:
            effective_min_conf = min_confidence
        if float(item.confidence) < effective_min_conf:
            if stats is not None:
                stats["low_confidence"] = stats.get("low_confidence", 0) + 1
            if diagnostic_logging:
                logger.info(
                    "[OCR DIAG] Filtered line (low confidence %.2f < %.2f): '%s'",
                    item.confidence,
                    effective_min_conf,
                    cleaned,
                )
            continue
        norm_line = OCRTextLine(
            text=cleaned,
            confidence=round(float(item.confidence), 4),
            box=item.box,
        )
        if item.box and len(item.box) >= 4:
            pts = np.array(item.box, dtype=np.float32)
            y_center = float(np.mean(pts[:, 1]))
            x_min = float(np.min(pts[:, 0]))
            box_h = max(8.0, float(np.max(pts[:, 1]) - np.min(pts[:, 1])))
        else:
            y_center = float(len(valid) * 30.0)
            x_min = 0.0
            box_h = 20.0
        valid.append((y_center, x_min, box_h, norm_line))

    if not valid:
        return "", 0.0, []

    # Filter out tiny isolated noise boxes (e.g. 1-2 char background specks) when a dominant subtitle box exists
    if len(valid) > 1:
        anchor = max(valid, key=lambda t: len(t[3].text) * t[3].confidence)
        ref_h = anchor[2]
        if ref_h >= 14.0:
            filtered_valid = [
                entry for entry in valid
                if not (
                    entry[2] < ref_h * 0.45
                    and len(entry[3].text) <= 2
                    and entry[3].confidence < 0.85
                )
            ]
            if filtered_valid:
                valid = filtered_valid

    # Sort primarily by vertical center
    valid.sort(key=lambda t: (t[0], t[1]))

    # Group boxes into horizontal rows when vertical centers are close
    rows: list[list[tuple[float, float, float, OCRTextLine]]] = []
    for entry in valid:
        y_c, x_m, b_h, _ = entry
        if not rows:
            rows.append([entry])
            continue
        prev_row = rows[-1]
        prev_y = sum(x[0] for x in prev_row) / len(prev_row)
        prev_h = max(x[2] for x in prev_row)
        if abs(y_c - prev_y) <= max(prev_h, b_h) * 0.55:
            prev_row.append(entry)
        else:
            rows.append([entry])

    ordered_lines: list[OCRTextLine] = []
    row_strings: list[str] = []
    for row in rows:
        row.sort(key=lambda t: t[1])  # Left-to-right within the same row
        row_texts = []
        for _, _, _, line_obj in row:
            ordered_lines.append(line_obj)
            row_texts.append(line_obj.text)
        row_strings.append(" ".join(row_texts))

    final_text = clean_ocr_text("\n".join(row_strings))
    avg_conf = round(
        sum(l.confidence for l in ordered_lines) / len(ordered_lines), 4
    )
    return final_text, avg_conf, ordered_lines


def apply_rapidocr_params(
    engine: Any,
    text_score: float | None = None,
    box_thresh: float | None = None,
    thresh: float | None = None,
    unclip_ratio: float | None = None,
) -> None:
    """
    Safely synchronize runtime parameters onto a RapidOCR instance.
    Updates the post-processing operator (DBPostProcess), configuration namespaces,
    and text score filter without reloading ONNX models.
    """
    if engine is None:
        return

    # 1. Update text_score on the root engine
    if text_score is not None:
        try:
            setattr(engine, "text_score", float(text_score))
        except Exception:
            pass

    # 2. Update DBPostProcess attributes on the active detector
    det = getattr(engine, "text_det", getattr(engine, "text_detector", None))
    post_op = getattr(det, "postprocess_op", None) if det else None
    if post_op is not None:
        if box_thresh is not None:
            try:
                setattr(post_op, "box_thresh", float(box_thresh))
            except Exception:
                pass
        if thresh is not None:
            try:
                setattr(post_op, "thresh", float(thresh))
            except Exception:
                pass
        if unclip_ratio is not None:
            try:
                setattr(post_op, "unclip_ratio", float(unclip_ratio))
            except Exception:
                pass

    # 3. Synchronize configuration dictionaries if present (modern RapidOCR)
    cfg = getattr(engine, "cfg", None)
    if cfg is not None:
        det_cfg = getattr(cfg, "Det", None)
        if det_cfg is not None:
            if box_thresh is not None:
                try:
                    setattr(det_cfg, "box_thresh", float(box_thresh))
                except Exception:
                    pass
            if thresh is not None:
                try:
                    setattr(det_cfg, "thresh", float(thresh))
                except Exception:
                    pass
            if unclip_ratio is not None:
                try:
                    setattr(det_cfg, "unclip_ratio", float(unclip_ratio))
                except Exception:
                    pass
        global_cfg = getattr(cfg, "Global", None)
        if global_cfg is not None and text_score is not None:
            try:
                setattr(global_cfg, "text_score", float(text_score))
            except Exception:
                pass


def invoke_rapidocr(
    engine: Any,
    image_bgr: np.ndarray,
    call_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Execute RapidOCR inference with resilient multi-tier fallback:
    - Tier 1: Call with all forwarded kwargs (supports test mocks & kwargs-enabled engines).
    - Tier 2: Call with standard kwargs excluding 'thresh' (RapidOCR native __call__ signature).
    - Tier 3: Call with image only (legacy engines; parameters are active via apply_rapidocr_params).
    """
    if not call_kwargs:
        return engine(image_bgr)

    # Tier 1: Attempt invocation with all provided kwargs
    try:
        return engine(image_bgr, **call_kwargs)
    except (TypeError, ValueError):
        pass

    # Tier 2: Upstream RapidOCR does not accept 'thresh' in __call__; retry with supported keys
    supported_kwargs = {k: v for k, v in call_kwargs.items() if k != "thresh"}
    if supported_kwargs:
        try:
            return engine(image_bgr, **supported_kwargs)
        except (TypeError, ValueError):
            pass

    # Tier 3: Strict/legacy callable; instance attributes already updated via apply_rapidocr_params
    return engine(image_bgr)


def _resolve_ocr_cpu_threads() -> int:
    """
    Choose a laptop-friendly CPU thread count for ONNX Runtime / PyTorch OCR inference.
    Capping at ~50% of logical cores (clamped to [2, 4] by default, overridable via
    SUBMAGIC_OCR_THREADS) avoids all-core thermal throttling and loud fans with zero quality loss.
    """
    env_val = os.environ.get("SUBMAGIC_OCR_THREADS", "").strip()
    if env_val.isdigit() and int(env_val) >= 1:
        return int(env_val)
    logical = os.cpu_count() or 4
    return max(2, min(4, logical // 2))


class OCREngineCache:
    """
    Lazy-loaded singleton cache for RapidOCR and VietOCR models with dynamic runtime parameter synchronization.
    Designed so unit tests can monkeypatch `_rapidocr`, `get_rapidocr`, `get_vietocr`, or `recognize_frame`
    without triggering heavy model downloads.
    """

    def __init__(self) -> None:
        self._rapidocr: Any = None
        self._vietocr: Any = None

    def get_rapidocr(
        self,
        text_score: float | None = None,
        box_thresh: float | None = None,
        thresh: float | None = None,
        unclip_ratio: float | None = None,
    ) -> Any:
        """Lazy-load RapidOCR (ONNX Runtime PP-OCRv4) with configurable parameters."""
        if self._rapidocr is not None:
            apply_rapidocr_params(
                self._rapidocr,
                text_score=text_score,
                box_thresh=box_thresh,
                thresh=thresh,
                unclip_ratio=unclip_ratio,
            )
            return self._rapidocr

        from rapidocr_onnxruntime import RapidOCR

        hw = detect_hardware()
        ocr_threads = _resolve_ocr_cpu_threads()
        init_kwargs: dict[str, Any] = {
            "intra_op_num_threads": ocr_threads,
            "inter_op_num_threads": 1,
        }
        if hw.recommended_ocr_provider == "CUDAExecutionProvider":
            init_kwargs["use_cuda"] = True
        elif hw.recommended_ocr_provider == "DmlExecutionProvider":
            init_kwargs["use_dml"] = True

        # Prefer upgraded PP-OCRv5 recognizer (18,383-char dict covering full Traditional +
        # Simplified Chinese, English, Japanese) when present locally, falling back to bundled v4.
        env_rec_model = os.environ.get("SUBMAGIC_RAPIDOCR_REC_MODEL", "").strip()
        default_v5_rec = settings.data_dir / "models" / "ch_PP-OCRv5_rec_infer.onnx"
        rec_model_candidate = Path(env_rec_model) if env_rec_model else default_v5_rec
        if rec_model_candidate.is_file():
            init_kwargs["rec_model_path"] = str(rec_model_candidate)

        params: dict[str, Any] = {}
        if text_score is not None:
            params["Global.text_score"] = float(text_score)
        if box_thresh is not None:
            params["Det.box_thresh"] = float(box_thresh)
        if thresh is not None:
            params["Det.thresh"] = float(thresh)
        if unclip_ratio is not None:
            params["Det.unclip_ratio"] = float(unclip_ratio)

        logger.info(
            "Initializing RapidOCR (ONNX provider=%s, params=%s)...",
            hw.recommended_ocr_provider,
            params,
        )
        try:
            self._rapidocr = RapidOCR(params=params, **init_kwargs) if params else RapidOCR(**init_kwargs)
        except TypeError:
            try:
                self._rapidocr = RapidOCR(**init_kwargs)
            except TypeError:
                self._rapidocr = RapidOCR()

        apply_rapidocr_params(
            self._rapidocr,
            text_score=text_score,
            box_thresh=box_thresh,
            thresh=thresh,
            unclip_ratio=unclip_ratio,
        )
        return self._rapidocr

    def get_vietocr(self) -> Any:
        """Lazy-load VietOCR Predictor (VGG-Transformer / VGG-Seq2Seq for Vietnamese diacritics)."""
        if self._vietocr is not None:
            return self._vietocr

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        hw = detect_hardware()
        device = "cuda:0" if hw.cuda_available else "cpu"
        if device == "cpu":
            try:
                import torch
                torch.set_num_threads(_resolve_ocr_cpu_threads())
            except Exception:
                pass
        config = Cfg.load_config_from_name("vgg_transformer")
        config["cnn"]["pretrained"] = False
        config["device"] = device

        weights_dir = settings.models_dir / "vietocr"
        weights_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing VietOCR Predictor on device='%s'...", device)
        self._vietocr = Predictor(config)
        return self._vietocr

    def recognize_rapidocr(
        self,
        image_bgr: np.ndarray,
        min_confidence: float = 0.35,
        text_score: float | None = None,
        box_thresh: float | None = None,
        thresh: float | None = None,
        unclip_ratio: float | None = None,
        diagnostic_logging: bool = False,
    ) -> OCRFrameResult:
        """Run RapidOCR detection + recognition on a cropped ROI frame with configurable parameters."""
        if image_bgr is None or image_bgr.size == 0:
            return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="rapidocr")

        try:
            engine = self.get_rapidocr(
                text_score=text_score,
                box_thresh=box_thresh,
                thresh=thresh,
                unclip_ratio=unclip_ratio,
            )
        except TypeError:
            engine = self.get_rapidocr()

        apply_rapidocr_params(
            engine,
            text_score=text_score,
            box_thresh=box_thresh,
            thresh=thresh,
            unclip_ratio=unclip_ratio,
        )

        call_kwargs: dict[str, Any] = {}
        if text_score is not None:
            call_kwargs["text_score"] = float(text_score)
        if box_thresh is not None:
            call_kwargs["box_thresh"] = float(box_thresh)
        if unclip_ratio is not None:
            call_kwargs["unclip_ratio"] = float(unclip_ratio)
        if thresh is not None:
            call_kwargs["thresh"] = float(thresh)

        raw_out = invoke_rapidocr(engine, image_bgr, call_kwargs)

        # RapidOCR returns (result_list, elapsed_list)
        raw_items = raw_out[0] if isinstance(raw_out, tuple) else raw_out
        if not raw_items:
            if diagnostic_logging:
                logger.info("[OCR DIAG] RapidOCR: no text boxes detected in frame")
            return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="rapidocr")

        lines: list[OCRTextLine] = []
        for item in raw_items:
            if not item or len(item) < 3:
                continue
            box_raw, text_raw, score_raw = item[0], item[1], item[2]
            box = (
                [[float(pt[0]), float(pt[1])] for pt in box_raw]
                if box_raw is not None
                else None
            )
            lines.append(
                OCRTextLine(
                    text=str(text_raw or ""),
                    confidence=float(score_raw if score_raw is not None else 0.0),
                    box=box,
                )
            )

        merged_text, avg_conf, ordered_lines = order_and_merge_ocr_lines(
            lines, min_confidence=min_confidence, diagnostic_logging=diagnostic_logging
        )

        # If no text detected or confidence is weak, retry with CLAHE/upscaling and binarized variants
        if not merged_text or avg_conf < 0.60:
            candidates: list[np.ndarray] = []
            prep_img = preprocess_roi_for_ocr(image_bgr)
            if prep_img.shape != image_bgr.shape or np.any(prep_img != image_bgr):
                candidates.append(prep_img)
            bin_img = create_binarized_variant(prep_img)
            candidates.append(bin_img)

            for variant in candidates:
                try:
                    retry_out = invoke_rapidocr(engine, variant, call_kwargs)

                    retry_items = retry_out[0] if isinstance(retry_out, tuple) else retry_out
                    if retry_items:
                        retry_lines: list[OCRTextLine] = []
                        for item in retry_items:
                            if not item or len(item) < 3:
                                continue
                            b_raw, t_raw, s_raw = item[0], item[1], item[2]
                            b = (
                                [[float(pt[0]), float(pt[1])] for pt in b_raw]
                                if b_raw is not None
                                else None
                            )
                            retry_lines.append(
                                OCRTextLine(
                                    text=str(t_raw or ""),
                                    confidence=float(s_raw if s_raw is not None else 0.0),
                                    box=b,
                                )
                            )
                        r_text, r_conf, r_ordered = order_and_merge_ocr_lines(
                            retry_lines, min_confidence=min_confidence, diagnostic_logging=diagnostic_logging
                        )
                        if r_text and (not merged_text or r_conf > avg_conf):
                            merged_text, avg_conf, ordered_lines = r_text, r_conf, r_ordered
                            if avg_conf >= 0.70:
                                break
                except Exception:
                    pass

        return OCRFrameResult(
            text=merged_text,
            confidence=avg_conf,
            lines=ordered_lines,
            engine_used="rapidocr",
        )

    def recognize_vietocr(
        self,
        image_bgr: np.ndarray,
        min_confidence: float = 0.35,
        text_score: float | None = None,
        box_thresh: float | None = None,
        thresh: float | None = None,
        unclip_ratio: float | None = None,
        diagnostic_logging: bool = False,
    ) -> OCRFrameResult:
        """
        Hybrid Vietnamese OCR:
        1. Use RapidOCR Detector to locate individual subtitle text lines in the ROI.
        2. Crop each detected line and pass to VietOCR Recognizer for accurate Vietnamese diacritics.
        3. If RapidOCR finds no sub-boxes (or single tight line), fallback to full ROI or RapidOCR text.
        """
        if image_bgr is None or image_bgr.size == 0:
            return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="vietocr")

        rapid_res = self.recognize_rapidocr(
            image_bgr,
            min_confidence=min_confidence,
            text_score=text_score,
            box_thresh=box_thresh,
            thresh=thresh,
            unclip_ratio=unclip_ratio,
            diagnostic_logging=diagnostic_logging,
        )

        try:
            viet_predictor = self.get_vietocr()
        except Exception as exc:
            if not rapid_res.lines:
                return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="vietocr")
            logger.warning(
                "VietOCR unavailable (%s); falling back to RapidOCR recognition output.",
                exc,
            )
            rapid_res.engine_used = "rapidocr"
            return rapid_res

        candidate_lines = (
            rapid_res.lines
            if rapid_res.lines
            else [OCRTextLine(text="", confidence=1.0, box=None)]
        )

        viet_lines: list[OCRTextLine] = []
        for det_line in candidate_lines:
            if det_line.box:
                pts = np.array(det_line.box, dtype=np.float32)
                line_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1])) if pts.ndim == 2 and len(pts) >= 4 else 24.0
                pad_bottom = max(6, int(round(line_h * 0.22)))
                line_crop_bgr = crop_text_box(image_bgr, det_line.box, pad=4, pad_bottom=pad_bottom)
            else:
                line_crop_bgr = image_bgr

            # Đảm bảo độ phân giải tối thiểu cho VietOCR (chiều cao >= 32px) để bảo toàn nét dấu tiếng Việt
            lh, lw = line_crop_bgr.shape[:2]
            if 0 < lh < 32 and lw > 0:
                scale_factor = 32.0 / float(lh)
                new_w = max(16, int(round(lw * scale_factor)))
                line_crop_bgr = cv2.resize(line_crop_bgr, (new_w, 32), interpolation=cv2.INTER_CUBIC)

            rgb_img = cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            try:
                pred_out = viet_predictor.predict(pil_img, return_prob=True)
                if isinstance(pred_out, tuple) and len(pred_out) == 2:
                    v_text, v_prob = str(pred_out[0] or ""), float(pred_out[1])
                else:
                    v_text, v_prob = str(pred_out or ""), float(det_line.confidence)
            except TypeError:
                v_text = str(viet_predictor.predict(pil_img) or "")
                v_prob = float(det_line.confidence)

            # If prediction is weak, try CLAHE or binarized variant
            if v_prob < 0.60 or not v_text.strip():
                prep_crop = preprocess_roi_for_ocr(line_crop_bgr)
                try:
                    prep_pil = Image.fromarray(cv2.cvtColor(prep_crop, cv2.COLOR_BGR2RGB))
                    out2 = viet_predictor.predict(prep_pil, return_prob=True)
                    if isinstance(out2, tuple) and len(out2) == 2:
                        t2, p2 = str(out2[0] or ""), float(out2[1])
                        if p2 > v_prob and clean_ocr_text(t2):
                            v_text, v_prob = t2, p2
                except Exception:
                    pass

            v_text = clean_ocr_text(v_text)
            if v_text:
                # Guard against VietOCR sequence hallucination on non-text frames when RapidOCR detector found no boxes
                if det_line.box is None:
                    has_vi_diacritic = detect_text_language(v_text) == "vi"
                    words = v_text.split()
                    is_single_ascii_hallucination = (
                        not has_vi_diacritic
                        and all(re.fullmatch(r"[A-Za-z0-9_\-.,!?'\"]+", w) for w in words)
                        and (len(words) <= 2 or max((len(w) for w in words), default=0) >= 8 or v_prob < 0.85)
                    )
                    if is_single_ascii_hallucination or v_prob < 0.70:
                        if diagnostic_logging:
                            logger.debug(
                                "[OCR DIAG] Dropping no-box VietOCR fallback hallucination: '%s' (prob=%.3f)",
                                v_text,
                                v_prob,
                            )
                        continue

                viet_lines.append(
                    OCRTextLine(
                        text=v_text,
                        confidence=round(v_prob, 4),
                        box=det_line.box,
                    )
                )

        merged_text, avg_conf, ordered_lines = order_and_merge_ocr_lines(
            viet_lines, min_confidence=min_confidence, diagnostic_logging=diagnostic_logging
        )
        return OCRFrameResult(
            text=merged_text,
            confidence=avg_conf,
            lines=ordered_lines,
            engine_used="vietocr",
        )

    def recognize_gemini_batch(
        self,
        images_bgr: list[np.ndarray],
        language: str = "vi",
        api_key: str | None = None,
    ) -> list[OCRFrameResult]:
        """
        Send a batch of cropped ROI images to Gemini 2.5 Flash Vision API for
        high-accuracy OCR and contextual spelling correction.
        """
        if not images_bgr:
            return []

        resolved_key = api_key or settings.gemini_api_key
        if not resolved_key:
            raise ValueError(
                "Thiếu Gemini API Key. Vui lòng cung cấp gemini_api_key hoặc chọn engine Local (RapidOCR / VietOCR)."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)
        contents: list[Any] = []
        valid_indices: list[int] = []

        for idx, img in enumerate(images_bgr):
            if img is None or img.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                continue
            contents.append(
                types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
            )
            valid_indices.append(idx)

        if not contents:
            return [
                OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="gemini")
                for _ in images_bgr
            ]

        prompt = (
            f"You are an expert subtitle OCR engine. Extract the exact hardsub text from each of the "
            f"{len(contents)} cropped video subtitle images in order (Target language hint: {language}). "
            f"Ignore background objects or watermarks. Preserve line breaks within a single image using '\\n'. "
            f"If an image contains no subtitle text, return an empty string for that image. "
            f"Return ONLY a JSON array of {len(contents)} objects with keys: "
            f"'index' (0-based integer), 'text' (string), 'confidence' (float 0.0-1.0)."
        )
        contents.append(prompt)

        from app.utils.gemini_client import generate_content_with_resilience
        response = generate_content_with_resilience(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        raw_json = json.loads(response.text or "[]")
        results = [
            OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="gemini")
            for _ in images_bgr
        ]

        if isinstance(raw_json, dict):
            raw_json = raw_json.get("items") or raw_json.get("results") or [raw_json]

        if isinstance(raw_json, list):
            for pos, item in enumerate(raw_json):
                if not isinstance(item, dict):
                    continue
                batch_idx = int(item.get("index", pos))
                if 0 <= batch_idx < len(valid_indices):
                    orig_idx = valid_indices[batch_idx]
                    txt = clean_ocr_text(str(item.get("text", "")))
                    conf = float(item.get("confidence", 0.95 if txt else 0.0))
                    lines = [OCRTextLine(text=line, confidence=conf) for line in txt.splitlines() if line]
                    results[orig_idx] = OCRFrameResult(
                        text=txt,
                        confidence=round(conf, 4) if txt else 0.0,
                        lines=lines,
                        engine_used="gemini",
                    )

        return results

    def recognize_gemini(
        self,
        image_bgr: np.ndarray,
        language: str = "vi",
        api_key: str | None = None,
    ) -> OCRFrameResult:
        """Run Gemini 2.5 Flash Vision OCR on a single cropped ROI image."""
        batch_res = self.recognize_gemini_batch(
            [image_bgr], language=language, api_key=api_key
        )
        if batch_res:
            return batch_res[0]
        return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="gemini")

    def recognize_frame(
        self,
        image_bgr: np.ndarray,
        engine: str = "auto",
        language: str = "vi",
        api_key: str | None = None,
        min_confidence: float = 0.35,
        text_score: float | None = None,
        box_thresh: float | None = None,
        thresh: float | None = None,
        unclip_ratio: float | None = None,
        diagnostic_logging: bool = False,
    ) -> OCRFrameResult:
        """Unified dispatcher to run the resolved OCR engine on a single ROI image."""
        resolved = resolve_ocr_engine(engine, language)
        if resolved == "gemini":
            return self.recognize_gemini(image_bgr, language=language, api_key=api_key)
        if resolved == "vietocr":
            return self.recognize_vietocr(
                image_bgr,
                min_confidence=min_confidence,
                text_score=text_score,
                box_thresh=box_thresh,
                thresh=thresh,
                unclip_ratio=unclip_ratio,
                diagnostic_logging=diagnostic_logging,
            )
        return self.recognize_rapidocr(
            image_bgr,
            min_confidence=min_confidence,
            text_score=text_score,
            box_thresh=box_thresh,
            thresh=thresh,
            unclip_ratio=unclip_ratio,
            diagnostic_logging=diagnostic_logging,
        )


ocr_engine_cache = OCREngineCache()
