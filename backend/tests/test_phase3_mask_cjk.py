"""
Milestone 3 Comprehensive Test Suite: Mask Coverage, Adaptive Thresholding & CJK Deduplication
Tests:
- Feature 11: Mask Coverage Verification (> 1 line height definite overflow, >= 0.75 suspected overflow, noise filter)
- Feature 12: Multi-Tier Remediation Ladder (Tier 1 sensitive+padding, Tier 2 edge strip OCR & geometric merge, Tier 3 zero degradation)
- Feature 13: Contrast-Gated Hybrid Percentile-Otsu Adaptive Thresholding (dark dim, bright outline, dark/bright empty, white flash)
- Feature 14: Relaxed Stroke Density Range [0.001, 0.85] with verified stroke core filter
- Feature 15: CJK Character-Level Strict Deduplication (7+ chars differing by 1 char, no artificial spaces, sub-7 tokenization)
- Feature 16: Mask Visual Difference Guard (dissimilarity >= 0.25 strictly blocks merge)
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from rapidfuzz import fuzz

from app.schemas import OCRJobRequest, SubtitleSegment
from app.services.ocr_engines import (
    MaskCoverageResult,
    OCRFrameResult,
    OCRTextLine,
    ocr_edge_strip_and_merge,
    recover_truncated_text_lines,
    remediate_mask_overflow,
    verify_mask_coverage,
)
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    compute_mask_visual_diff,
    deduplicate_ocr_segments,
    extract_meaningful_cjk_characters,
    extract_text_binary_mask,
    has_text_activity,
    is_cjk_string,
    tokenize_for_dedup,
)


# ==============================================================================
# Feature 11 & 12: Mask Coverage Verification & Multi-Tier Remediation
# ==============================================================================


class TestMaskCoverageVerificationAndRemediation:
    """Tests for Feature 11 (Coverage Verification) and Feature 12 (Remediation Ladder)."""

    def test_verify_mask_coverage_definite_and_suspected_overflow(self):
        """
        Verify definite overflow (Delta >= 1.0 * H_line) and suspected overflow (Delta >= 0.75 * H_line),
        while filtering noise (Delta < 0.25 * H_line or Delta <= 3px).
        """
        mask_h, mask_w = 60, 400
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        # Line height = 24px (y=18..42)
        line_h = 24
        # Mask text strokes extend from x=50 to x=350
        mask[18:42, 50:350] = 255

        # 1. Definite left overflow: OCR box starts at x=75 (Delta = 25px > 1.0 * 24px)
        box_definite = [[75, 18], [350, 18], [350, 42], [75, 42]]
        lines_def = [OCRTextLine(text="DEFINITE OVERFLOW", confidence=0.9, box=box_definite)]
        res_def = verify_mask_coverage(mask, lines_def, min_overflow_ratio=0.75)
        assert res_def.has_overflow is True
        assert res_def.overflow_dir == "left"
        assert res_def.overflow_px == 25
        assert res_def.overflow_px >= line_h

        # 2. Suspected left overflow: OCR box starts at x=69 (Delta = 19px, where 0.75*24=18 <= 19 < 24)
        box_suspect = [[69, 18], [350, 18], [350, 42], [69, 42]]
        lines_sus = [OCRTextLine(text="SUSPECTED OVERFLOW", confidence=0.9, box=box_suspect)]
        res_sus = verify_mask_coverage(mask, lines_sus, min_overflow_ratio=0.75)
        assert res_sus.has_overflow is True
        assert res_sus.overflow_dir == "left"
        assert res_sus.overflow_px == 19

        # 3. Noise rejection: Delta = 5px (5 < 0.25 * 24 = 6px) -> noise, must be ignored
        box_noise = [[55, 18], [350, 18], [350, 42], [55, 42]]
        lines_noise = [OCRTextLine(text="NOISE OVERFLOW", confidence=0.9, box=box_noise)]
        res_noise = verify_mask_coverage(mask, lines_noise, min_overflow_ratio=0.75)
        assert res_noise.has_overflow is False

        # 4. Tiny noise <= 3px -> must be ignored
        box_tiny = [[52, 18], [350, 18], [350, 42], [52, 42]]
        lines_tiny = [OCRTextLine(text="TINY NOISE", confidence=0.9, box=box_tiny)]
        res_tiny = verify_mask_coverage(mask, lines_tiny, min_overflow_ratio=0.75)
        assert res_tiny.has_overflow is False

    def test_verify_mask_coverage_multiline_union(self):
        """Verify multiline subtitle boxes are properly unified across horizontal bounds."""
        mask = np.zeros((90, 500), dtype=np.uint8)
        # 2-line subtitle: line 1 (y=10..35, x=100..400), line 2 (y=45..70, x=50..450)
        mask[10:35, 100:400] = 255
        mask[45:70, 50:450] = 255

        # OCR detected line 1 at x=100..400, line 2 clipped at x=90..450 (left overflow 40px)
        lines = [
            OCRTextLine(text="Line One", box=[[100, 10], [400, 10], [400, 35], [100, 35]]),
            OCRTextLine(text="Line Two", box=[[90, 45], [450, 45], [450, 70], [90, 70]]),
        ]

        # Overall mask union is x=50..450. OCR union box starts at min(100, 90) = 90.
        # Overflow is 90 - 50 = 40px > avg line_h (25px).
        cov = verify_mask_coverage(mask, lines, min_overflow_ratio=0.75)
        assert cov.has_overflow is True
        assert cov.overflow_dir == "left"
        assert cov.overflow_px == 40

    def test_tier1_remediation_border_padding_at_frame_edge(self):
        """Tier 1: Border replicate padding enables recovery when text touches the frame boundary x=0."""
        frame = np.full((60, 500, 3), 40, dtype=np.uint8)
        mask = np.zeros((60, 500), dtype=np.uint8)
        # Text starts at x=0 (touching left boundary)
        mask[15:45, 0:350] = 255

        # Initial detector missed the leftmost 35px at the image border
        initial_line = OCRTextLine(
            box=[[35, 15], [350, 15], [350, 45], [35, 45]],
            text="REMAINDER OF BORDER SUBTITLE",
            confidence=0.85,
        )
        initial_res = OCRFrameResult(
            text=initial_line.text, confidence=0.85, lines=[initial_line]
        )

        mock_recognizer = MagicMock()
        recovered_line = OCRTextLine(
            box=[[0, 15], [350, 15], [350, 45], [0, 45]],
            text="EDGE_TOKEN REMAINDER OF BORDER SUBTITLE",
            confidence=0.93,
        )
        mock_recognizer.return_value = OCRFrameResult(
            text=recovered_line.text, confidence=0.93, lines=[recovered_line]
        )

        remediated = recover_truncated_text_lines(
            image_bgr=frame,
            roi_mask=mask,
            ocr_res=initial_res,
            recognize_fn=mock_recognizer,
            retry_box_thresh=0.25,
            retry_unclip_ratio=2.2,
        )

        assert mock_recognizer.call_count == 1
        assert remediated.text == "EDGE_TOKEN REMAINDER OF BORDER SUBTITLE"
        assert remediated.confidence == 0.93

    def test_tier2_edge_strip_isolated_recovery_and_cjk_stitching(self):
        """
        Tier 2: When Tier 1 full strip retry does not recover new characters,
        Tier 2 isolates the edge strip, runs independent OCR, and geometrically merges prefix.
        Zero hardcoded strings.
        """
        frame = np.full((60, 500, 3), 40, dtype=np.uint8)
        mask = np.zeros((60, 500), dtype=np.uint8)
        mask[15:45, 20:380] = 255

        initial_line = OCRTextLine(
            box=[[60, 15], [380, 15], [380, 45], [60, 45]],
            text="後續內容示例",
            confidence=0.88,
        )
        initial_res = OCRFrameResult(
            text=initial_line.text, confidence=0.88, lines=[initial_line]
        )

        # Recognizer mock:
        # Tier 1 call (full expanded strip) fails to find additional text (returns same text)
        # Tier 2 call (isolated edge strip) successfully recognizes the missing leading CJK ideogram
        tier_calls = {"count": 0}

        def mock_recognizer(strip, **kwargs):
            tier_calls["count"] += 1
            if tier_calls["count"] == 1:
                # Tier 1 returns same text (no improvement)
                return OCRFrameResult(text="後續內容示例", confidence=0.88, lines=[initial_line])
            # Tier 2 (isolated edge crop) finds the missing leading character
            return OCRFrameResult(
                text="首字",
                confidence=0.92,
                lines=[OCRTextLine(text="首字", confidence=0.92, box=[[0, 0], [40, 0], [40, 30], [0, 30]])],
            )

        remediated = recover_truncated_text_lines(
            image_bgr=frame,
            roi_mask=mask,
            ocr_res=initial_res,
            recognize_fn=mock_recognizer,
        )

        assert tier_calls["count"] >= 2, "Tier 2 edge strip recognizer must be triggered"
        assert remediated.text == "首字後續內容示例"
        assert remediated.confidence >= 0.90

    def test_tier2_edge_strip_overlap_deduplication(self):
        """Tier 2: If the isolated edge strip contains an overlapping boundary character, deduplicate it."""
        frame = np.full((60, 500, 3), 40, dtype=np.uint8)
        initial_line = OCRTextLine(
            box=[[60, 15], [380, 15], [380, 45], [60, 45]],
            text="月神教",
            confidence=0.88,
        )
        initial_res = OCRFrameResult(text="月神教", confidence=0.88, lines=[initial_line])

        # Edge crop recognized "日月" (where '月' overlaps with start of '月神教')
        mock_rec = MagicMock()
        mock_rec.return_value = OCRFrameResult(
            text="日月",
            confidence=0.91,
            lines=[OCRTextLine(text="日月", confidence=0.91, box=[[0, 0], [30, 0], [30, 30], [0, 30]])],
        )

        merged_res = ocr_edge_strip_and_merge(
            image_bgr=frame,
            ocr_res=initial_res,
            edge_crop_coords=(10, 15, 65, 45),
            direction="left",
            recognize_fn=mock_rec,
        )

        assert merged_res is not None
        assert merged_res.text == "日月神教", "Overlapping '月' must be deduplicated"

    def test_tier3_zero_degradation_guard_on_unrecoverable_noise(self):
        """Tier 3: When neither Tier 1 nor Tier 2 recover additional characters, original result is preserved."""
        frame = np.full((60, 500, 3), 40, dtype=np.uint8)
        mask = np.zeros((60, 500), dtype=np.uint8)
        mask[15:45, 20:380] = 255

        initial_line = OCRTextLine(
            box=[[80, 15], [380, 15], [380, 45], [80, 45]],
            text="ORIGINAL SUBTITLE",
            confidence=0.90,
        )
        initial_res = OCRFrameResult(
            text=initial_line.text, confidence=0.90, lines=[initial_line]
        )

        # Mock that returns empty text or raises error
        failing_rec = MagicMock(return_value=OCRFrameResult(text="", confidence=0.0, lines=[]))

        remediated = recover_truncated_text_lines(
            image_bgr=frame,
            roi_mask=mask,
            ocr_res=initial_res,
            recognize_fn=failing_rec,
        )

        assert remediated == initial_res
        assert remediated.text == "ORIGINAL SUBTITLE"
        assert remediated.confidence == 0.90

    def test_zero_hardcoded_benchmark_strings_in_ast(self):
        """Integrity check: Zero hardcoded benchmark strings or words in source code."""
        forbidden_tokens = ["東方不敗", "誰能想到", "日月神教大教主", "天下風雲出我輩"]
        files_to_check = [
            Path("app/services/ocr_engines.py"),
            Path("app/services/ocr_pipeline.py"),
        ]
        for rel_path in files_to_check:
            full_path = Path("backend") / rel_path if (Path("backend") / rel_path).exists() else rel_path
            assert full_path.exists(), f"Source file {full_path} not found"
            content = full_path.read_text(encoding="utf-8")
            # Parse AST to ensure valid Python syntax
            ast.parse(content)
            for token in forbidden_tokens:
                assert token not in content, f"Integrity violation: forbidden benchmark token '{token}' in {rel_path}"


# ==============================================================================
# Feature 13 & 14: Adaptive ROI Thresholding & Relaxed Stroke Density
# ==============================================================================


class TestAdaptiveROIThresholdingAndStrokeDensity:
    """Tests for Feature 13 (Contrast-Gated Hybrid Percentile-Otsu) and Feature 14 (Stroke Density 0.85)."""

    def test_adaptive_threshold_dark_dim_subtitles(self):
        """Feature 13: Dark scene with dim/gray text (luma=145 on bg=35, below 185) is captured."""
        roi = np.full((60, 300, 3), 35, dtype=np.uint8)
        # Draw dim text with luminance ~145
        cv2.putText(roi, "DIM TEXT 145", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (145, 145, 145), 2)

        mask_adaptive = extract_text_binary_mask(roi, adaptive=True)
        assert has_text_activity(mask_adaptive) is True
        assert cv2.countNonZero(mask_adaptive) > 50

        # Contrast with fixed threshold 185: fails to capture dim text
        mask_fixed = extract_text_binary_mask(roi, luma_threshold=185, adaptive=False)
        assert has_text_activity(mask_fixed) is False
        assert cv2.countNonZero(mask_fixed) == 0

    def test_adaptive_threshold_bright_scene_with_black_outline(self):
        """Feature 13: Bright scene (bg=200) with white text (255) and black outline (20) is captured."""
        roi = np.full((60, 300, 3), 200, dtype=np.uint8)
        # Black outline
        cv2.putText(roi, "BRIGHT SCENE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4)
        # White fill
        cv2.putText(roi, "BRIGHT SCENE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        mask = extract_text_binary_mask(roi, adaptive=True)
        assert has_text_activity(mask) is True
        assert cv2.countNonZero(mask) > 50

    def test_adaptive_threshold_dark_empty_scene_suppressed(self):
        """Feature 13: Dark empty scene (bg=35 with noise) has no text activity."""
        roi = np.full((60, 300, 3), 35, dtype=np.uint8)
        mask = extract_text_binary_mask(roi, adaptive=True)
        assert has_text_activity(mask) is False

    def test_adaptive_threshold_bright_empty_scene_suppressed(self):
        """Feature 13: Bright empty scene (bg=200) has no text activity."""
        roi = np.full((60, 300, 3), 200, dtype=np.uint8)
        mask = extract_text_binary_mask(roi, adaptive=True)
        assert has_text_activity(mask) is False

    def test_adaptive_threshold_white_flash_suppressed(self):
        """Feature 13: Full-frame white flash (255) is suppressed immediately."""
        flash = np.full((60, 300, 3), 255, dtype=np.uint8)
        mask = extract_text_binary_mask(flash, adaptive=True)
        assert has_text_activity(mask) is False

    def test_adaptive_threshold_bright_texture_noise_suppressed(self):
        """Feature 13: Bright scene with surface texture (luma in [180, 215], contrast < 15) is early-exited."""
        noise_roi = np.full((60, 300, 3), 200, dtype=np.uint8)
        # Add low contrast texture (+/- 5 luma)
        rng = np.random.default_rng(42)
        texture = rng.integers(-5, 6, size=(60, 300, 3), dtype=np.int16)
        noise_roi = np.clip(noise_roi.astype(np.int16) + texture, 0, 255).astype(np.uint8)

        mask = extract_text_binary_mask(noise_roi, adaptive=True)
        assert has_text_activity(mask) is False
        assert cv2.countNonZero(mask) == 0

    def test_relaxed_stroke_density_accepts_thick_subtitle_strokes(self):
        """Feature 14: Relaxed upper bound 0.85 accepts genuine thick subtitle strokes (density=72%)."""
        # Create solid text block strokes with width >= 3-4px (e.g. bold 2-line subtitle)
        mask = np.zeros((100, 400), dtype=np.uint8)
        mask[12:87, 20:380] = 255

        ratio = cv2.countNonZero(mask) / float(mask.size)
        assert 0.65 < ratio <= 0.85, f"Density {ratio:.3f} should be in (0.65, 0.85]"

        # Must accept with max_ratio=0.85
        assert has_text_activity(mask, max_ratio=0.85) is True

    def test_relaxed_stroke_density_rejects_fragmented_texture_noise(self):
        """Feature 14: High-density fragmented noise (density=72%) is rejected by erosion core filter."""
        # Create fine 1-pixel checkerboard/speckle noise
        mask = np.zeros((100, 400), dtype=np.uint8)
        rng = np.random.default_rng(123)
        # 75% non-zero random single pixels
        random_noise = (rng.random((100, 400)) < 0.72).astype(np.uint8) * 255
        ratio = cv2.countNonZero(random_noise) / float(random_noise.size)
        assert 0.65 < ratio <= 0.85

        # Must reject because 1px noise evaporates under 3x3 erosion
        assert has_text_activity(random_noise, max_ratio=0.85) is False


# ==============================================================================
# Feature 15 & 16: CJK Deduplication & Mask Visual Difference Guard
# ==============================================================================


class TestCJKDeduplicationAndMaskVisualGuard:
    """Tests for Feature 15 (CJK Strict Dedup) and Feature 16 (Mask Visual Diff Guard)."""

    def test_cjk_seven_characters_differ_by_one_character_must_not_merge(self):
        """Feature 15: 7-character CJK sentences differing by 1 char (NO artificial spaces) must NOT merge."""
        cjk_a = "日月神教大教主"
        cjk_b = "日月神教副教主"

        # Standard Levenshtein ratio is 85.71% (exceeds 85% threshold)
        assert fuzz.ratio(cjk_a, cjk_b) > 85.0

        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=cjk_a)
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=cjk_b)

        dedup.add_or_merge(s1)
        _, merged = dedup.add_or_merge(s2)

        assert merged is False, f"Falsely merged distinct CJK sentences '{cjk_a}' and '{cjk_b}'"
        assert len(dedup.segments) == 2
        assert dedup.segments[0].text == cjk_a
        assert dedup.segments[1].text == cjk_b

    def test_cjk_eight_characters_differ_by_one_character_must_not_merge(self):
        """Feature 15: 8-character CJK sentences (Levenshtein 87.5%) differing by 1 char must NOT merge."""
        cjk1 = "天下風雲出我輩"
        cjk2 = "天下風雲出他輩"

        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=cjk1)
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=cjk2)

        dedup.add_or_merge(s1)
        _, merged = dedup.add_or_merge(s2)

        assert merged is False
        assert len(dedup.segments) == 2
        assert dedup.segments[0].text == cjk1
        assert dedup.segments[1].text == cjk2

    def test_cjk_identical_sentences_merge_and_extend_timing(self):
        """Feature 15: Consecutive identical CJK sentences must merge and span the full duration."""
        cjk = "誰能想到東方不敗"
        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=cjk, confidence=0.88)
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.2, text=cjk, confidence=0.96)

        dedup.add_or_merge(s1)
        res, merged = dedup.add_or_merge(s2)

        assert merged is True
        assert len(dedup.segments) == 1
        assert dedup.segments[0].start_time == 1.0
        assert dedup.segments[0].end_time == 3.2
        assert dedup.segments[0].confidence == 0.96

    def test_cjk_punctuation_variation_merges_gracefully(self):
        """Feature 15: Trailing punctuation differences should merge if meaningful characters match."""
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="日月神教大教主，")
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="日月神教大教主")

        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
        dedup.add_or_merge(s1)
        _, merged = dedup.add_or_merge(s2)

        assert merged is True
        assert len(dedup.segments) == 1

    def test_cjk_sub_7_characters_protected_by_tokenization(self):
        """Feature 15: 5-character CJK string differing by 1 char is protected via character tokenization."""
        cjk_5a = "東方不敗好"
        cjk_5b = "東方必敗好"

        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=80.0, max_merge_gap_sec=0.20)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=cjk_5a)
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=cjk_5b)

        dedup.add_or_merge(s1)
        _, merged = dedup.add_or_merge(s2)

        assert merged is False, f"5-character CJK differing by 1 char must not merge: {cjk_5a} vs {cjk_5b}"
        assert len(dedup.segments) == 2

    def test_mask_visual_diff_guard_blocks_merge_when_diff_exceeds_threshold(self):
        """Feature 16: Mask Visual Difference Guard blocks merging when diff >= 0.25, even if text matches."""
        # Top subtitle mask vs Bottom subtitle mask (disjoint)
        mask_top = np.zeros((100, 400), dtype=np.uint8)
        mask_top[10:35, 50:350] = 255

        mask_bottom = np.zeros((100, 400), dtype=np.uint8)
        mask_bottom[65:90, 50:350] = 255

        diff = compute_mask_visual_diff(mask_top, mask_bottom)
        assert diff >= 0.25, f"Expected visual diff >= 0.25, got {diff}"

        dedup = FuzzySubtitleDeduplicator(mask_visual_diff_threshold=0.25)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="日月神教大教主")
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="日月神教大教主")

        dedup.add_or_merge(s1, mask=mask_top)
        _, merged = dedup.add_or_merge(s2, mask=mask_bottom)

        assert merged is False, "Visually disjoint masks must block merge even on identical text"
        assert len(dedup.segments) == 2

    def test_mask_visual_diff_guard_allows_merge_when_diff_is_low(self):
        """Feature 16: Mask Visual Difference Guard allows merge when masks visually match (diff < 0.25)."""
        mask1 = np.zeros((100, 400), dtype=np.uint8)
        mask1[20:50, 50:350] = 255

        mask2 = np.zeros((100, 400), dtype=np.uint8)
        mask2[21:50, 50:350] = 255  # 1-pixel vertical shift

        diff = compute_mask_visual_diff(mask1, mask2)
        assert diff < 0.25, f"Expected visual diff < 0.25, got {diff}"

        dedup = FuzzySubtitleDeduplicator(mask_visual_diff_threshold=0.25)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="日月神教大教主")
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="日月神教大教主")

        dedup.add_or_merge(s1, mask=mask1)
        _, merged = dedup.add_or_merge(s2, mask=mask2)

        assert merged is True
        assert len(dedup.segments) == 1

    def test_mask_visual_guard_o1_memory_and_none_fallback(self):
        """Feature 16: Deduplicator stores only _last_mask in O(1) memory and handles mask=None gracefully."""
        dedup = FuzzySubtitleDeduplicator()
        assert dedup._last_mask is None

        m1 = np.full((30, 100), 255, dtype=np.uint8)
        s1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="Subtitle A")
        dedup.add_or_merge(s1, mask=m1)
        assert dedup._last_mask is m1

        # Fallback with mask=None
        s2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="Subtitle B")
        dedup.add_or_merge(s2, mask=None)
        assert dedup._last_mask is None
        assert len(dedup.segments) == 2

    def test_cjk_punctuation_semantic_equality_and_merge(self):
        """Feature 15 Remediation: CJK sentences identical in semantic characters merge despite punctuation variations."""
        test_cases = [
            ("你好，世界！", "你好世界", "Short 4-char CJK with comma and exclamation"),
            ("你好，世界", "你好世界", "Short 4-char CJK with single comma"),
            ("你好世界！", "你好世界", "Short 4-char CJK with single exclamation"),
            ("天下風雲出我輩，！", "天下風雲出我輩", "7-char CJK with two punctuation marks ，！"),
            ("你好世界", "你好，世界！", "Reverse: plain then punctuated"),
            ("天下風雲出我輩", "天下風雲出我輩，！", "Reverse: 7-char plain then punctuated"),
        ]

        for s_first, s_second, desc in test_cases:
            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_first, confidence=0.90)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_second, confidence=0.95)

            dedup.add_or_merge(seg1)
            res, merged = dedup.add_or_merge(seg2)

            assert merged is True, f"Failed to merge {desc}: '{s_first}' vs '{s_second}'"
            assert len(dedup.segments) == 1
            assert res.start_time == 1.0
            assert res.end_time == 3.0
            assert res.confidence == 0.95


class TestCJKNoiseCleanupAndFragmentBridge:
    """Tests for CJK edge-noise cleanup, auto-mode majority filter, and adjacent fragment bridge."""

    def test_clean_cjk_subtitle_text_strips_edge_noise_and_preserves_units(self):
        from app.services.ocr_pipeline import clean_cjk_subtitle_text

        assert clean_cjk_subtitle_text("5死了") == "死了"
        assert clean_cjk_subtitle_text("1你没死") == "你没死"
        assert clean_cjk_subtitle_text("啊 DDN") == "啊"
        assert clean_cjk_subtitle_text("你們了星 S") == "你們了星"
        assert clean_cjk_subtitle_text("是你抓走了盈盈B") == "是你抓走了盈盈"
        assert clean_cjk_subtitle_text("令狐公子 子") == "令狐公子"
        assert clean_cjk_subtitle_text("别 來 無 恙 啊") == "别來無恙啊"
        assert clean_cjk_subtitle_text("哈哈哈 哈哈") == "哈哈哈哈哈"
        # Preserve valid numeric expressions and non-CJK text
        assert clean_cjk_subtitle_text("2024年") == "2024年"
        assert clean_cjk_subtitle_text("50万") == "50万"
        assert clean_cjk_subtitle_text("Hello world 123") == "Hello world 123"

    def test_filter_cjk_majority_noise_segments_in_auto_mode(self):
        from app.services.ocr_pipeline import filter_cjk_majority_noise_segments

        segs = [
            SubtitleSegment(id="1", project_id="p", index=1, start_time=0.0, end_time=1.0, text="令狐大侠"),
            SubtitleSegment(id="2", project_id="p", index=2, start_time=1.0, end_time=2.0, text="YA"),
            SubtitleSegment(id="3", project_id="p", index=3, start_time=2.0, end_time=3.0, text="任盈盈你也没興趣"),
            SubtitleSegment(id="4", project_id="p", index=4, start_time=3.0, end_time=4.0, text="ULUDE 茶 喉"),
            SubtitleSegment(id="5", project_id="p", index=5, start_time=4.0, end_time=5.0, text="是或不是"),
            SubtitleSegment(id="6", project_id="p", index=6, start_time=5.0, end_time=6.0, text="哈哈哈哈哈"),
        ]
        kept, dropped = filter_cjk_majority_noise_segments(segs, language="auto")
        assert [s.text for s in kept] == ["令狐大侠", "任盈盈你也没興趣", "是或不是", "哈哈哈哈哈"]
        assert [s.text for s in dropped] == ["YA", "ULUDE 茶 喉"]
        assert [s.index for s in kept] == [1, 2, 3, 4]

    def test_is_same_cjk_subtitle_fragment_bridges_noise_but_keeps_distinct_sentences(self):
        from app.services.ocr_pipeline import is_same_cjk_subtitle_fragment

        assert is_same_cjk_subtitle_fragment("扌據是被令狐大侠殺了", "A 扌據是被令狐大侠殺了") is True
        assert is_same_cjk_subtitle_fragment("希望我是主", "吴希望我是") is True
        assert is_same_cjk_subtitle_fragment("真點", "言真點") is True
        assert is_same_cjk_subtitle_fragment("女如你所", "如 口你所") is True
        # Distinct 7-char sentences differing by an interior character must NOT bridge
        assert is_same_cjk_subtitle_fragment("日月神教大教主", "日月神教副教主") is False


