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
) -> np.ndarray:
    """
    Crop a 4-point text bounding polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    from `image_bgr` with perspective rectification (or padded bounding rect fallback).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((16, 64, 3), dtype=np.uint8)

    h, w = image_bgr.shape[:2]
    pts = np.array(box, dtype=np.float32)
    if pts.shape != (4, 2):
        x_coords = pts[:, 0] if pts.ndim == 2 else [0, w]
        y_coords = pts[:, 1] if pts.ndim == 2 else [0, h]
        x1 = max(0, int(np.floor(np.min(x_coords))) - pad)
        y1 = max(0, int(np.floor(np.min(y_coords))) - pad)
        x2 = min(w, int(np.ceil(np.max(x_coords))) + pad)
        y2 = min(h, int(np.ceil(np.max(y_coords))) + pad)
        if x2 <= x1 or y2 <= y1:
            return image_bgr.copy()
        return image_bgr[y1:y2, x1:x2].copy()

    # Axis-aligned padded crop for nearly horizontal subtitle boxes (faster and avoids resampling blur)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = max(0, int(np.floor(np.min(xs))) - pad)
    y1 = max(0, int(np.floor(np.min(ys))) - pad)
    x2 = min(w, int(np.ceil(np.max(xs))) + pad)
    y2 = min(h, int(np.ceil(np.max(ys))) + pad)

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
        effective_min_conf = (
            max(0.20, min_confidence - 0.10)
            if (is_cjk or len(cleaned) <= 4)
            else min_confidence
        )
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


class OCREngineCache:
    """
    Lazy-loaded singleton cache for RapidOCR and VietOCR models.
    Designed so unit tests can monkeypatch `get_rapidocr`, `get_vietocr`, or `recognize_frame`
    without triggering heavy model downloads.
    """

    def __init__(self) -> None:
        self._rapidocr: Any = None
        self._vietocr: Any = None

    def get_rapidocr(self) -> Any:
        """Lazy-load RapidOCR (ONNX Runtime PP-OCRv4)."""
        if self._rapidocr is not None:
            return self._rapidocr

        from rapidocr_onnxruntime import RapidOCR

        hw = detect_hardware()
        kwargs: dict[str, Any] = {}
        if hw.recommended_ocr_provider == "CUDAExecutionProvider":
            kwargs["use_cuda"] = True
        elif hw.recommended_ocr_provider == "DmlExecutionProvider":
            kwargs["use_dml"] = True

        logger.info(
            "Initializing RapidOCR (ONNX provider=%s)...",
            hw.recommended_ocr_provider,
        )
        try:
            self._rapidocr = RapidOCR(**kwargs)
        except TypeError:
            self._rapidocr = RapidOCR()
        return self._rapidocr

    def get_vietocr(self) -> Any:
        """Lazy-load VietOCR Predictor (VGG-Transformer / VGG-Seq2Seq for Vietnamese diacritics)."""
        if self._vietocr is not None:
            return self._vietocr

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        hw = detect_hardware()
        device = "cuda:0" if hw.cuda_available else "cpu"
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
    ) -> OCRFrameResult:
        """Run RapidOCR detection + recognition on a cropped ROI frame."""
        if image_bgr is None or image_bgr.size == 0:
            return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="rapidocr")

        engine = self.get_rapidocr()
        raw_out = engine(image_bgr)
        # RapidOCR returns (result_list, elapsed_list)
        raw_items = raw_out[0] if isinstance(raw_out, tuple) else raw_out
        if not raw_items:
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
            lines, min_confidence=min_confidence
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
                    retry_out = engine(variant)
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
                            retry_lines, min_confidence=min_confidence
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
    ) -> OCRFrameResult:
        """
        Hybrid Vietnamese OCR:
        1. Use RapidOCR Detector to locate individual subtitle text lines in the ROI.
        2. Crop each detected line and pass to VietOCR Recognizer for accurate Vietnamese diacritics.
        3. If RapidOCR finds no sub-boxes (or single tight line), fallback to full ROI or RapidOCR text.
        """
        if image_bgr is None or image_bgr.size == 0:
            return OCRFrameResult(text="", confidence=0.0, lines=[], engine_used="vietocr")

        rapid_res = self.recognize_rapidocr(image_bgr, min_confidence=min_confidence)

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
                line_crop_bgr = crop_text_box(image_bgr, det_line.box, pad=4)
            else:
                line_crop_bgr = image_bgr

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
                viet_lines.append(
                    OCRTextLine(
                        text=v_text,
                        confidence=round(v_prob, 4),
                        box=det_line.box,
                    )
                )

        merged_text, avg_conf, ordered_lines = order_and_merge_ocr_lines(
            viet_lines, min_confidence=min_confidence
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
    ) -> OCRFrameResult:
        """Unified dispatcher to run the resolved OCR engine on a single ROI image."""
        resolved = resolve_ocr_engine(engine, language)
        if resolved == "gemini":
            return self.recognize_gemini(image_bgr, language=language, api_key=api_key)
        if resolved == "vietocr":
            return self.recognize_vietocr(image_bgr, min_confidence=min_confidence)
        return self.recognize_rapidocr(image_bgr, min_confidence=min_confidence)


ocr_engine_cache = OCREngineCache()
