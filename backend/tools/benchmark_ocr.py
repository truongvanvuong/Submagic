"""Benchmark script to measure OCR pipeline throughput and accuracy."""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from rapidfuzz import fuzz

from app.services.ocr_engines import ocr_engine_cache, clean_ocr_text


def create_test_subtitle_frame(text: str, width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a realistic video frame with moving noise background and subtitle text."""
    # Gradient background with slight noise
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        frame[y, :, :] = (int(y / height * 120), int(40 + y / height * 80), int(150 - y / height * 50))
    noise = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Subtitle text area at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (width - tw) // 2
    ty = height - 60

    # Draw black outline + white text
    cv2.putText(frame, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Return cropped ROI (bottom 18%)
    y1 = int(height * 0.80)
    return frame[y1:height, :]


def benchmark():
    test_phrases = [
        "SUB MAGIC AUTOMATED HARD SUBTITLE SCANNER",
        "AI POWERED REAL TIME VIDEO EXTRACTION",
        "FAST ACCURATE OPEN CV FILTERED RECOGNITION",
        "TIENG VIET SUBTITLE CHUAN XAC CAO",
        "HELLO WORLD TESTING PIPELINE PERFORMANCE",
    ]

    print("=============================================================")
    print("      SUB MAGIC OCR PIPELINE BENCHMARK (BEFORE VS AFTER)     ")
    print("=============================================================")

    # Warm up engine
    sample_crop = create_test_subtitle_frame("WARMUP ENGINE")
    ocr_engine_cache.recognize_frame(sample_crop, engine="rapidocr", language="auto")

    # 1. Baseline: Raw Full Resolution Crop
    start_raw = time.perf_counter()
    raw_results = []
    for phrase in test_phrases:
        crop = create_test_subtitle_frame(phrase)
        res = ocr_engine_cache.recognize_frame(crop, engine="rapidocr", language="auto")
        raw_results.append(clean_ocr_text(res.text))
    time_raw = time.perf_counter() - start_raw

    # 2. Optimized: Pre-scaled to 64px height with aspect-ratio preservation
    start_opt = time.perf_counter()
    opt_results = []
    for phrase in test_phrases:
        crop = create_test_subtitle_frame(phrase)
        if crop.shape[0] > 72:
            scale = 64.0 / float(crop.shape[0])
            new_w = max(16, int(round(crop.shape[1] * scale)))
            crop = cv2.resize(crop, (new_w, 64), interpolation=cv2.INTER_AREA)
        res = ocr_engine_cache.recognize_frame(crop, engine="rapidocr", language="auto")
        opt_results.append(clean_ocr_text(res.text))
    time_opt = time.perf_counter() - start_opt

    # Calculate accuracy comparing against original phrases
    raw_acc = np.mean([fuzz.ratio(p.upper(), r.upper()) for p, r in zip(test_phrases, raw_results)])
    opt_acc = np.mean([fuzz.ratio(p.upper(), r.upper()) for p, r in zip(test_phrases, opt_results)])

    speedup = ((time_raw - time_opt) / max(0.001, time_raw)) * 100.0

    print(f"\n[Raw Baseline]    Time: {time_raw:.3f}s ({len(test_phrases)/time_raw:.1f} FPS) | Accuracy: {raw_acc:.1f}%")
    print(f"[Optimized 64px]  Time: {time_opt:.3f}s ({len(test_phrases)/time_opt:.1f} FPS) | Accuracy: {opt_acc:.1f}%")
    print(f"-> Speed improvement: {speedup:+.1f}% | Accuracy delta: {opt_acc - raw_acc:+.1f}%")
    print("=============================================================\n")


if __name__ == "__main__":
    benchmark()
