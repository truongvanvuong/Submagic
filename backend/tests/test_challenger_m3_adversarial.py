"""
Empirical Adversarial Challenge Test Harness for Milestone 3
===========================================================
Targets:
- Feature 11 & 12: Mask Coverage Verification & Remediation Ladder (Left/Right Overflow & Noise Suppression)
- Feature 13: Contrast-Gated Hybrid Percentile-Otsu Adaptive Thresholding (Early Exit & Bright/Dark Contrast)
- Feature 14: Relaxed Stroke Density Range [0.001, 0.85] (Solid Strokes vs Fragmented Noise)

Executes directly via:
    uv run python backend/tests/test_challenger_m3_adversarial.py
or via pytest:
    uv run --directory backend pytest tests/test_challenger_m3_adversarial.py -v
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.services.ocr_engines import (
    MaskCoverageResult,
    OCRFrameResult,
    OCRTextLine,
    ocr_edge_strip_and_merge,
    recover_truncated_text_lines,
    verify_mask_coverage,
)
from app.services.ocr_pipeline import (
    extract_text_binary_mask,
    has_text_activity,
)


def log_test_header(title: str):
    print(f"\n{'='*70}\n[TEST] {title}\n{'='*70}")


# ==============================================================================
# Challenge 1: Mask Coverage Overflow & Multi-Tier Remediation (Features 11 & 12)
# ==============================================================================

def test_challenge_left_overflow_detection_and_remediation():
    log_test_header("Challenge 1A: Left Edge Overflow (> 1.0 Line Height)")
    # Image: 80x600 BGR, line height = 30px (y=20..50)
    img_h, img_w = 80, 600
    line_h = 30
    frame = np.full((img_h, img_w, 3), 30, dtype=np.uint8)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Subtitle mask extends horizontally from x=35 to x=450
    mask[20:50, 35:450] = 255

    # OCR detection box clipped on the left: starts at x=80 (Delta = 80 - 35 = 45px > 1.0 * line_h = 30px)
    clipped_box = [[80, 20], [450, 20], [450, 50], [80, 50]]
    orig_text = "後續內容示例"
    orig_line = OCRTextLine(text=orig_text, confidence=0.88, box=clipped_box)
    orig_res = OCRFrameResult(text=orig_text, confidence=0.88, lines=[orig_line])

    # 1. Verify detection
    cov = verify_mask_coverage(mask, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Coverage detection: has_overflow={cov.has_overflow}, dir={cov.overflow_dir}, px={cov.overflow_px}")
    assert cov.has_overflow is True, "Must detect overflow"
    assert cov.overflow_dir == "left", "Overflow direction must be 'left'"
    assert cov.overflow_px == 45, f"Expected 45px overflow, got {cov.overflow_px}"
    assert cov.overflow_px > 1.0 * line_h, "Overflow must strictly exceed 1.0 line height"

    # 2. Verify Tier 1 (Sensitive retry on expanded crop)
    rec_tier1 = MagicMock()
    recovered_text_tier1 = "完整的前綴後續內容示例"
    rec_tier1.return_value = OCRFrameResult(
        text=recovered_text_tier1,
        confidence=0.94,
        lines=[OCRTextLine(text=recovered_text_tier1, confidence=0.94, box=[[35, 20], [450, 20], [450, 50], [35, 50]])],
    )
    res_tier1 = recover_truncated_text_lines(
        image_bgr=frame,
        roi_mask=mask,
        ocr_res=orig_res,
        recognize_fn=rec_tier1,
        min_overflow_ratio=0.75,
    )
    print(f"[-] Tier 1 result: text='{res_tier1.text}', calls={rec_tier1.call_count}")
    assert rec_tier1.call_count == 1, "Tier 1 must call recognizer exactly once"
    assert res_tier1.text == recovered_text_tier1, "Tier 1 must accept expanded text"
    assert res_tier1.confidence == 0.94

    # 3. Verify Tier 2 (Isolated edge strip OCR & prefix stitching when Tier 1 fails)
    tier2_calls = []
    def mock_tier2_recognizer(strip_img, **kwargs):
        tier2_calls.append(strip_img.shape)
        if len(tier2_calls) == 1:
            # Tier 1 call returns same clipped text (no expansion)
            return OCRFrameResult(text=orig_text, confidence=0.88, lines=[orig_line])
        # Tier 2 call on isolated edge strip returns prefix token
        return OCRFrameResult(
            text="前綴",
            confidence=0.91,
            lines=[OCRTextLine(text="前綴", confidence=0.91, box=[[0, 0], [40, 0], [40, 30], [0, 30]])],
        )

    res_tier2 = recover_truncated_text_lines(
        image_bgr=frame,
        roi_mask=mask,
        ocr_res=orig_res,
        recognize_fn=mock_tier2_recognizer,
        min_overflow_ratio=0.75,
    )
    print(f"[-] Tier 2 result: text='{res_tier2.text}', calls={len(tier2_calls)}")
    assert len(tier2_calls) == 2, "Must execute Tier 1 then Tier 2"
    assert res_tier2.text == "前綴後續內容示例", f"Expected stitched prefix, got '{res_tier2.text}'"
    assert res_tier2.confidence == 0.91

    # 4. Verify Tier 2 Prefix Overlap Deduplication (1 char and 2 chars)
    # 1-char overlap: edge strip = "ABC", original = "CDE" -> "ABCDE"
    mock_overlap1 = MagicMock(return_value=OCRFrameResult(
        text="ABC", confidence=0.90,
        lines=[OCRTextLine(text="ABC", confidence=0.90, box=[[0, 0], [30, 0], [30, 30], [0, 30]])]
    ))
    res_overlap1 = ocr_edge_strip_and_merge(
        image_bgr=frame,
        ocr_res=OCRFrameResult(text="CDE", confidence=0.85, lines=[orig_line]),
        edge_crop_coords=(10, 20, 90, 50),
        direction="left",
        recognize_fn=mock_overlap1,
    )
    assert res_overlap1 is not None and res_overlap1.text == "ABCDE", f"Expected 'ABCDE', got '{res_overlap1.text}'"

    # 2-char overlap: edge strip = "ABCD", original = "CDXYZ" -> "ABCDXYZ"
    mock_overlap2 = MagicMock(return_value=OCRFrameResult(
        text="ABCD", confidence=0.90,
        lines=[OCRTextLine(text="ABCD", confidence=0.90, box=[[0, 0], [40, 0], [40, 30], [0, 30]])]
    ))
    res_overlap2 = ocr_edge_strip_and_merge(
        image_bgr=frame,
        ocr_res=OCRFrameResult(text="CDXYZ", confidence=0.85, lines=[orig_line]),
        edge_crop_coords=(10, 20, 90, 50),
        direction="left",
        recognize_fn=mock_overlap2,
    )
    assert res_overlap2 is not None and res_overlap2.text == "ABCDXYZ", f"Expected 'ABCDXYZ', got '{res_overlap2.text}'"
    print("[+] Left overflow tests passed successfully.")


def test_challenge_right_overflow_detection_and_remediation():
    log_test_header("Challenge 1B: Right Edge Overflow (> 1.0 Line Height)")
    img_h, img_w = 80, 600
    line_h = 30
    frame = np.full((img_h, img_w, 3), 30, dtype=np.uint8)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Subtitle mask extends horizontally from x=50 to x=550
    mask[20:50, 50:550] = 255

    # OCR detection box clipped on the right: ends at x=505 (Delta = 550 - 505 = 45px > 1.0 * line_h = 30px)
    clipped_box = [[50, 20], [505, 20], [505, 50], [50, 50]]
    orig_text = "前續內容示例"
    orig_line = OCRTextLine(text=orig_text, confidence=0.88, box=clipped_box)
    orig_res = OCRFrameResult(text=orig_text, confidence=0.88, lines=[orig_line])

    # 1. Verify detection
    cov = verify_mask_coverage(mask, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Coverage detection: has_overflow={cov.has_overflow}, dir={cov.overflow_dir}, px={cov.overflow_px}")
    assert cov.has_overflow is True
    assert cov.overflow_dir == "right"
    assert cov.overflow_px == 45
    assert cov.overflow_px > 1.0 * line_h

    # 2. Verify Tier 1 Sensitive Retry Suffix Expansion
    rec_tier1 = MagicMock()
    recovered_text_tier1 = "前續內容示例末尾後綴"
    rec_tier1.return_value = OCRFrameResult(
        text=recovered_text_tier1,
        confidence=0.93,
        lines=[OCRTextLine(text=recovered_text_tier1, confidence=0.93, box=[[50, 20], [550, 20], [550, 50], [50, 50]])],
    )
    res_tier1 = recover_truncated_text_lines(
        image_bgr=frame,
        roi_mask=mask,
        ocr_res=orig_res,
        recognize_fn=rec_tier1,
        min_overflow_ratio=0.75,
    )
    print(f"[-] Tier 1 result: text='{res_tier1.text}', calls={rec_tier1.call_count}")
    assert rec_tier1.call_count == 1
    assert res_tier1.text == recovered_text_tier1

    # 3. Verify Tier 2 Suffix Stitching
    tier2_calls = []
    def mock_tier2_recognizer(strip_img, **kwargs):
        tier2_calls.append(strip_img.shape)
        if len(tier2_calls) == 1:
            return OCRFrameResult(text=orig_text, confidence=0.88, lines=[orig_line])
        return OCRFrameResult(
            text="尾字",
            confidence=0.92,
            lines=[OCRTextLine(text="尾字", confidence=0.92, box=[[0, 0], [40, 0], [40, 30], [0, 30]])],
        )

    res_tier2 = recover_truncated_text_lines(
        image_bgr=frame,
        roi_mask=mask,
        ocr_res=orig_res,
        recognize_fn=mock_tier2_recognizer,
        min_overflow_ratio=0.75,
    )
    print(f"[-] Tier 2 result: text='{res_tier2.text}', calls={len(tier2_calls)}")
    assert len(tier2_calls) == 2
    assert res_tier2.text == "前續內容示例尾字", f"Expected suffix stitched, got '{res_tier2.text}'"

    # 4. Verify Tier 2 Suffix Overlap Deduplication
    # 1-char overlap: original = "ABC", edge strip = "CDE" -> "ABCDE"
    mock_overlap1 = MagicMock(return_value=OCRFrameResult(
        text="CDE", confidence=0.90,
        lines=[OCRTextLine(text="CDE", confidence=0.90, box=[[0, 0], [30, 0], [30, 30], [0, 30]])]
    ))
    res_overlap1 = ocr_edge_strip_and_merge(
        image_bgr=frame,
        ocr_res=OCRFrameResult(text="ABC", confidence=0.85, lines=[orig_line]),
        edge_crop_coords=(490, 20, 560, 50),
        direction="right",
        recognize_fn=mock_overlap1,
    )
    assert res_overlap1 is not None and res_overlap1.text == "ABCDE", f"Expected 'ABCDE', got '{res_overlap1.text}'"

    # 2-char overlap: original = "ABCD", edge strip = "CDXYZ" -> "ABCDXYZ"
    mock_overlap2 = MagicMock(return_value=OCRFrameResult(
        text="CDXYZ", confidence=0.90,
        lines=[OCRTextLine(text="CDXYZ", confidence=0.90, box=[[0, 0], [40, 0], [40, 30], [0, 30]])]
    ))
    res_overlap2 = ocr_edge_strip_and_merge(
        image_bgr=frame,
        ocr_res=OCRFrameResult(text="ABCD", confidence=0.85, lines=[orig_line]),
        edge_crop_coords=(490, 20, 560, 50),
        direction="right",
        recognize_fn=mock_overlap2,
    )
    assert res_overlap2 is not None and res_overlap2.text == "ABCDXYZ", f"Expected 'ABCDXYZ', got '{res_overlap2.text}'"
    print("[+] Right overflow tests passed successfully.")


def test_challenge_noise_overflow_suppression():
    log_test_header("Challenge 1C: Noise Overflow Suppression (< 0.25 Line Height or <= 3px)")
    img_h, img_w = 80, 600
    line_h = 32
    frame = np.full((img_h, img_w, 3), 30, dtype=np.uint8)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Base OCR box: x=100..400
    base_box = [[100, 20], [400, 20], [400, 52], [100, 52]]
    orig_line = OCRTextLine(text="SUBTITLE NOISE TEST", confidence=0.90, box=base_box)
    orig_res = OCRFrameResult(text="SUBTITLE NOISE TEST", confidence=0.90, lines=[orig_line])

    # Case 1: Left Delta = 2px (<= 3px) -> Noise
    mask_2px = mask.copy()
    mask_2px[20:52, 98:400] = 255
    cov_2px = verify_mask_coverage(mask_2px, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Delta=2px: has_overflow={cov_2px.has_overflow}, px={cov_2px.overflow_px}")
    assert cov_2px.has_overflow is False, "Delta=2px (<=3px) must be rejected as noise"

    # Case 2: Left Delta = 3px (<= 3px boundary) -> Noise
    mask_3px = mask.copy()
    mask_3px[20:52, 97:400] = 255
    cov_3px = verify_mask_coverage(mask_3px, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Delta=3px: has_overflow={cov_3px.has_overflow}, px={cov_3px.overflow_px}")
    assert cov_3px.has_overflow is False, "Delta=3px (<=3px) must be rejected as noise"

    # Case 3: Left Delta = 6px (6px < 0.25 * 32px = 8px) -> Noise
    mask_6px = mask.copy()
    mask_6px[20:52, 94:400] = 255
    cov_6px = verify_mask_coverage(mask_6px, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Delta=6px (<0.25*32=8px): has_overflow={cov_6px.has_overflow}, px={cov_6px.overflow_px}")
    assert cov_6px.has_overflow is False, "Delta=6px (<0.25*H_line) must be rejected as noise"

    # Case 4: Right Delta = 7px (7px < 0.25 * 32px = 8px) -> Noise
    mask_right_7px = mask.copy()
    mask_right_7px[20:52, 100:407] = 255
    cov_right_7px = verify_mask_coverage(mask_right_7px, [orig_line], min_overflow_ratio=0.75)
    print(f"[-] Right Delta=7px (<8px): has_overflow={cov_right_7px.has_overflow}, px={cov_right_7px.overflow_px}")
    assert cov_right_7px.has_overflow is False, "Right Delta=7px must be rejected as noise"

    # Case 5: Zero Remediation Overhead Verification
    # Ensure recognizer is NEVER invoked when noise occurs
    mock_rec = MagicMock()
    for test_mask, label in [(mask_2px, "2px"), (mask_3px, "3px"), (mask_6px, "6px"), (mask_right_7px, "right 7px")]:
        mock_rec.reset_mock()
        res = recover_truncated_text_lines(
            image_bgr=frame,
            roi_mask=test_mask,
            ocr_res=orig_res,
            recognize_fn=mock_rec,
            min_overflow_ratio=0.75,
        )
        assert mock_rec.call_count == 0, f"Remediation recognizer must NOT be called for noise {label}"
        assert res == orig_res, f"Original result must be preserved without modification for {label}"
    print("[+] Noise suppression and zero overhead verified successfully.")


# ==============================================================================
# Challenge 2: Adaptive ROI Thresholding (Feature 13)
# ==============================================================================

def test_challenge_adaptive_threshold_flat_blank_frame_early_exit():
    log_test_header("Challenge 2A: Flat Blank Frame Early Exit (P95 - P50 < 15)")

    # 1. Solid black (luma 0), mid gray (luma 128), bright white (luma 255)
    for luma in [0, 40, 100, 128, 180, 220, 255]:
        roi = np.full((60, 300, 3), luma, dtype=np.uint8)
        mask = extract_text_binary_mask(roi, adaptive=True)
        assert mask.shape == (60, 300), "Mask shape must match input ROI"
        assert cv2.countNonZero(mask) == 0, f"Solid luma={luma} must return all zero mask"
        assert has_text_activity(mask) is False, f"Solid luma={luma} must have no text activity"

    # 2. Noisy flat frame (mean=150, noise in [-4, +4], P95 - P50 < 15)
    rng = np.random.default_rng(42)
    noise = rng.integers(-4, 5, size=(60, 300, 3), dtype=np.int16)
    noisy_roi = np.clip(150 + noise, 0, 255).astype(np.uint8)
    mask_noisy = extract_text_binary_mask(noisy_roi, adaptive=True)
    assert cv2.countNonZero(mask_noisy) == 0, "Noisy flat frame must trigger early exit"
    assert has_text_activity(mask_noisy) is False

    # 3. Performance Benchmark: Early exit latency vs Full processing
    # Early exit frame (solid 100)
    flat_frame = np.full((60, 300, 3), 100, dtype=np.uint8)
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = extract_text_binary_mask(flat_frame, adaptive=True)
    elapsed_flat = (time.perf_counter() - t0) / iters

    # Frame with text (requires full processing pipeline)
    text_frame = np.full((60, 300, 3), 40, dtype=np.uint8)
    cv2.putText(text_frame, "ACTIVE SUBTITLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = extract_text_binary_mask(text_frame, adaptive=True)
    elapsed_text = (time.perf_counter() - t0) / iters

    speedup = elapsed_text / max(1e-6, elapsed_flat)
    print(f"[-] Early exit latency: {elapsed_flat*1000:.3f} ms/frame")
    print(f"[-] Text frame latency: {elapsed_text*1000:.3f} ms/frame")
    print(f"[-] Early exit speedup: {speedup:.1f}x faster")
    assert elapsed_flat < 0.001, f"Early exit must be < 1.0 ms on CPU, got {elapsed_flat*1000:.3f} ms"
    assert speedup >= 2.0, f"Early exit should be substantially faster than full pipeline, got {speedup:.1f}x"
    print("[+] Flat blank frame early exit verified successfully.")


def test_challenge_adaptive_threshold_bright_and_dark_backgrounds():
    log_test_header("Challenge 2B: Bright Background (P50 >= 140) and Dark Background (P50 < 140)")

    # 1. Bright background (bg=210, P50 >= 140) with white text (255) and dark outline (20)
    bright_roi = np.full((60, 350, 3), 210, dtype=np.uint8)
    # Dark outline
    cv2.putText(bright_roi, "OUTLINED TEXT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4)
    # Bright fill
    cv2.putText(bright_roi, "OUTLINED TEXT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    mask_bright = extract_text_binary_mask(bright_roi, adaptive=True)
    assert has_text_activity(mask_bright) is True, "White text on bright background must be captured"
    nz_bright = cv2.countNonZero(mask_bright)
    print(f"[-] Bright background with outlined text captured: nz={nz_bright} px")
    assert nz_bright > 100, "Must capture significant stroke pixels"

    # Empty bright background (bg=210, no text) -> must be suppressed
    empty_bright = np.full((60, 350, 3), 210, dtype=np.uint8)
    mask_empty_bright = extract_text_binary_mask(empty_bright, adaptive=True)
    assert has_text_activity(mask_empty_bright) is False, "Empty bright scene must have no text activity"

    # 2. Dark background (bg=35, P50 < 140) with dim text (luma=145)
    # This was the core failure mode of fixed threshold 185!
    dark_roi = np.full((60, 350, 3), 35, dtype=np.uint8)
    cv2.putText(dark_roi, "DIM TEXT 145", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (145, 145, 145), 2)

    # Adaptive threshold captures it
    mask_dark_adaptive = extract_text_binary_mask(dark_roi, adaptive=True)
    assert has_text_activity(mask_dark_adaptive) is True, "Adaptive threshold must capture dim subtitles on dark background"
    nz_dark = cv2.countNonZero(mask_dark_adaptive)
    print(f"[-] Dark background dim text captured by adaptive: nz={nz_dark} px")
    assert nz_dark > 50

    # Fixed threshold 185 completely misses it
    mask_dark_fixed = extract_text_binary_mask(dark_roi, luma_threshold=185, adaptive=False)
    assert has_text_activity(mask_dark_fixed) is False, "Fixed threshold 185 misses dim text"
    assert cv2.countNonZero(mask_dark_fixed) == 0

    # 3. Very low contrast test on dark scene: text luma 70 on bg 20 (contrast = 50)
    low_contrast_roi = np.full((60, 350, 3), 20, dtype=np.uint8)
    cv2.putText(low_contrast_roi, "LOW CONTRAST 70", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
    mask_low = extract_text_binary_mask(low_contrast_roi, adaptive=True)
    assert has_text_activity(mask_low) is True, "Low contrast text (luma 70 on bg 20) must be captured by Otsu"
    print(f"[-] Low contrast (70 on 20) captured: nz={cv2.countNonZero(mask_low)} px")

    print("[+] Bright and dark background contrast tests passed successfully.")


# ==============================================================================
# Challenge 3: Stroke Density Relaxation (0.85) (Feature 14)
# ==============================================================================

def test_challenge_stroke_density_relaxation_thick_strokes_vs_noise():
    log_test_header("Challenge 3: Stroke Density (0.65, 0.85] - Solid Strokes vs Noise")
    h, w = 100, 400
    total_pixels = float(h * w)

    # Test 1: Thick Solid Subtitle Text in (0.65, 0.85]
    # Simulate thick bold subtitle blocks (e.g. 2-line bold CJK strokes)
    mask_thick = np.zeros((h, w), dtype=np.uint8)
    # Block stroke of height 72px, width 390px -> area = 28080 = 70.2%
    mask_thick[14:86, 5:395] = 255
    ratio_thick = cv2.countNonZero(mask_thick) / total_pixels
    print(f"[-] Thick subtitle density: {ratio_thick*100:.2f}% (in (0.65, 0.85])")
    assert 0.65 < ratio_thick <= 0.85

    # Core ratio calculation: 3x3 erosion
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_thick = cv2.erode(mask_thick, k3, iterations=1)
    core_thick = cv2.countNonZero(eroded_thick) / float(cv2.countNonZero(mask_thick))
    print(f"[-] Thick subtitle erosion core ratio R_core: {core_thick:.3f} (>= 0.12)")
    assert core_thick >= 0.12, "Solid strokes must retain substantial core after erosion"

    # Must be ACCEPTED with relaxed bound 0.85
    assert has_text_activity(mask_thick, max_ratio=0.85) is True, "Thick subtitle text must be ACCEPTED"

    # With the old fixed bound 0.65, it was REJECTED!
    assert has_text_activity(mask_thick, max_ratio=0.65) is False, "With max_ratio=0.65 it was rejected"

    # Test 2: High-Frequency Salt-and-Pepper Noise / Background Texture in (0.65, 0.85] where R_core < 0.12
    # Uncorrelated Bernoulli noise at p in (0.65, 0.75] produces R_core = p^8 < 0.12 (e.g. 0.68^8=0.046, 0.72^8=0.072)
    rng = np.random.default_rng(999)
    for target_density in [0.68, 0.72]:
        noise_mask = (rng.random((h, w)) < target_density).astype(np.uint8) * 255
        ratio_noise = cv2.countNonZero(noise_mask) / total_pixels
        assert 0.65 < ratio_noise <= 0.85
        eroded_noise = cv2.erode(noise_mask, k3, iterations=1)
        core_noise = cv2.countNonZero(eroded_noise) / float(cv2.countNonZero(noise_mask))
        print(f"[-] Salt-and-pepper density={ratio_noise*100:.2f}%, R_core={core_noise:.3f} (< 0.12)")
        assert core_noise < 0.12, f"Noise R_core must be < 0.12, got {core_noise:.3f}"
        assert has_text_activity(noise_mask, max_ratio=0.85) is False, f"Noise at {ratio_noise:.2f} must be REJECTED"

    # Test 3: Fine 2-pixel texture noise stripes (density = 66.0%)
    # 2px lines with 1px gap: leaves zero core (R_core = 0.0 < 0.12) under 3x3 erosion
    texture_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if y % 3 != 0:  # rows 1,2 filled, row 0 empty -> 2px bars (66/100 = 66.0%)
            texture_mask[y, :] = 255
    ratio_texture = cv2.countNonZero(texture_mask) / total_pixels
    print(f"[-] 2px texture noise density: {ratio_texture*100:.2f}% (in (0.65, 0.85])")
    assert 0.65 < ratio_texture <= 0.85
    eroded_texture = cv2.erode(texture_mask, k3, iterations=1)
    core_texture = cv2.countNonZero(eroded_texture) / float(cv2.countNonZero(texture_mask))
    print(f"[-] 2px texture erosion core ratio R_core: {core_texture:.3f} (< 0.12)")
    assert core_texture < 0.12, "2px texture noise core ratio must be < 0.12"
    assert has_text_activity(texture_mask, max_ratio=0.85) is False, "2px texture noise must be REJECTED"

    # Test 4: Density Boundary Checks
    # Exact boundary 0.85 vs 0.851
    mask_85 = np.zeros((100, 100), dtype=np.uint8)
    mask_85[:85, :] = 255  # 85% solid
    assert cv2.countNonZero(mask_85) / 10000.0 == 0.85
    assert has_text_activity(mask_85, max_ratio=0.85) is True, "85% solid stroke must be accepted"

    mask_86 = np.zeros((100, 100), dtype=np.uint8)
    mask_86[:86, :] = 255  # 86% solid
    assert cv2.countNonZero(mask_86) / 10000.0 == 0.86
    assert has_text_activity(mask_86, max_ratio=0.85) is False, "86% exceeds max_ratio=0.85 and must be rejected"

    # Tier 1 fast-path boundary: 0.65 is accepted without erosion check
    mask_65_noisy = (rng.random((100, 100)) < 0.645).astype(np.uint8) * 255
    assert cv2.countNonZero(mask_65_noisy) / 10000.0 <= 0.65
    assert has_text_activity(mask_65_noisy, max_ratio=0.85) is True, "<= 0.65 enters Tier 1 fast path"

    print("[+] Stroke density relaxation tests passed successfully.")


# ==============================================================================
# Runner for CLI invocation
# ==============================================================================

def main():
    print("=" * 70)
    print("STARTING EMPIRICAL ADVERSARIAL CHALLENGE SUITE (MILESTONE 3)")
    print("=" * 70)

    t_start = time.perf_counter()
    test_challenge_left_overflow_detection_and_remediation()
    test_challenge_right_overflow_detection_and_remediation()
    test_challenge_noise_overflow_suppression()
    test_challenge_adaptive_threshold_flat_blank_frame_early_exit()
    test_challenge_adaptive_threshold_bright_and_dark_backgrounds()
    test_challenge_stroke_density_relaxation_thick_strokes_vs_noise()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"ALL ADVERSARIAL CHALLENGES COMPLETED IN {total_time:.3f} SECONDS WITH ZERO FAILURES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
