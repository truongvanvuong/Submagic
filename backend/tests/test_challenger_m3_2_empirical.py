# -*- coding: utf-8 -*-
"""
Adversarial Empirical Challenge Test Suite for Milestone 3 (Challenger 2).

Scope:
- Feature 15: CJK Character-Level Deduplication
  * 7-char and 8-char CJK sentences differing by 1 char (MUST NOT merge)
  * CJK sentences with punctuation: '你好，世界！' vs '你好世界'
  * Sub-7 character CJK sentences differing by 1 char (MUST NOT merge)
  * Non-CJK strings (Vietnamese, English) Levenshtein similarity merging preserved
- Feature 16: Mask Visual Difference Guard
  * Identical text with subtitle position jump (diff >= 0.25, MUST block merge)
  * Identical text with slight frame jitter (diff < 0.25, MUST allow merge)
  * O(1) memory and None fallback
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.schemas import SubtitleSegment
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    compute_mask_visual_diff,
    compute_ocr_text_similarity,
    extract_meaningful_cjk_characters,
    is_cjk_string,
    normalize_for_fuzzy,
    tokenize_for_dedup,
)


class TestFeature15CJKStrictDeduplication(unittest.TestCase):
    """Empirical challenge tests for Feature 15 (CJK Strict Deduplication)."""

    def test_cjk_7_chars_differ_by_one_char_never_merge(self):
        """
        Adversarial Test: 7-character CJK sentences differing by exactly 1 character.
        Levenshtein similarity is 6/7 = 85.71% (> 85.0%).
        FuzzySubtitleDeduplicator MUST NEVER falsely merge them!
        """
        test_pairs = [
            ("日月神教大教主", "日月神教副教主"),
            ("天下風雲出我輩", "天下風雲出他輩"),
            ("笑傲江湖之東方", "笑傲江湖之西方"),
            ("誰能想到東方敗", "誰能想到西方敗"),
        ]

        for s_prev, s_cand in test_pairs:
            sim = compute_ocr_text_similarity(s_prev, s_cand)
            self.assertGreater(
                sim,
                85.0,
                f"Prerequisite check: '{s_prev}' vs '{s_cand}' should have similarity > 85%, got {sim:.2f}%",
            )

            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_prev)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_cand)

            dedup.add_or_merge(seg1)
            _, merged = dedup.add_or_merge(seg2)

            self.assertFalse(
                merged,
                f"REGRESSION: 7-character CJK sentences '{s_prev}' and '{s_cand}' were falsely merged!",
            )
            self.assertEqual(
                len(dedup.segments),
                2,
                f"Expected 2 distinct segments, but got {len(dedup.segments)}",
            )
            self.assertEqual(dedup.segments[0].text, s_prev)
            self.assertEqual(dedup.segments[1].text, s_cand)

    def test_cjk_8_chars_differ_by_one_char_never_merge(self):
        """
        Adversarial Test: 8-character CJK sentences differing by exactly 1 character.
        Levenshtein similarity is 7/8 = 87.5% (> 85.0%).
        FuzzySubtitleDeduplicator MUST NEVER falsely merge them!
        """
        test_pairs = [
            ("誰能想到東方不敗", "誰能想到西方不敗"),
            ("笑傲江湖之東方敗", "笑傲江湖之西方敗"),
            ("日月神教大教主在", "日月神教副教主在"),
        ]

        for s_prev, s_cand in test_pairs:
            sim = compute_ocr_text_similarity(s_prev, s_cand)
            self.assertGreater(
                sim,
                85.0,
                f"Prerequisite check: '{s_prev}' vs '{s_cand}' should have similarity > 85%, got {sim:.2f}%",
            )

            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_prev)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_cand)

            dedup.add_or_merge(seg1)
            _, merged = dedup.add_or_merge(seg2)

            self.assertFalse(
                merged,
                f"REGRESSION: 8-character CJK sentences '{s_prev}' and '{s_cand}' were falsely merged!",
            )
            self.assertEqual(len(dedup.segments), 2)
            self.assertEqual(dedup.segments[0].text, s_prev)
            self.assertEqual(dedup.segments[1].text, s_cand)

    def test_cjk_sub_7_chars_differ_by_one_char_never_merge(self):
        """
        Adversarial Test: CJK sentences < 7 chars differing by 1 char.
        Character-level tokenization prevents space-splitting word grouping bugs.
        """
        test_pairs = [
            ("東方不敗好", "東方必敗好"),  # 5 chars
            ("笑傲江湖好", "笑傲江湖美"),  # 5 chars
            ("天龍八部之一", "天龍八部之二"),  # 6 chars
        ]

        for s_prev, s_cand in test_pairs:
            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=80.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_prev)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_cand)

            dedup.add_or_merge(seg1)
            _, merged = dedup.add_or_merge(seg2)

            self.assertFalse(
                merged,
                f"REGRESSION: Short CJK sentences '{s_prev}' and '{s_cand}' were falsely merged!",
            )
            self.assertEqual(len(dedup.segments), 2)

    def test_cjk_sentences_with_punctuation_deduplication(self):
        """
        Adversarial Test: CJK sentences with punctuation variation.
        Dispatch test case: '你好，世界！' vs '你好世界'.
        Both represent the identical semantic subtitle text with/without punctuation.
        """
        test_cases = [
            ("你好，世界！", "你好世界", "Short 4-char CJK with comma and exclamation"),
            ("你好，世界", "你好世界", "Short 4-char CJK with single comma"),
            ("你好世界！", "你好世界", "Short 4-char CJK with single exclamation"),
            ("天下風雲出我輩，！", "天下風雲出我輩", "7-char CJK with two punctuation marks ，！"),
        ]

        failures = []
        for s_punct, s_plain, desc in test_cases:
            chars_punct = extract_meaningful_cjk_characters(s_punct)
            chars_plain = extract_meaningful_cjk_characters(s_plain)
            self.assertEqual(chars_punct, chars_plain)

            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_punct)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_plain)

            dedup.add_or_merge(seg1)
            _, merged = dedup.add_or_merge(seg2)

            if not merged:
                failures.append(f"{desc}: '{s_punct}' vs '{s_plain}' created {len(dedup.segments)} segments")

        self.assertFalse(
            failures,
            f"CHALLENGE FAILURE: CJK punctuation stripping failed to merge in {len(failures)} cases:\n"
            + "\n".join(f"  - {f}" for f in failures),
        )

    def test_non_cjk_strings_levenshtein_similarity_merging_intact(self):
        """
        Adversarial Test: Non-CJK strings (English and Vietnamese) must retain
        standard Levenshtein fuzzy merging when similarity >= 85%.
        """
        merge_pairs = [
            ("Hello welcome to Submagic", "Hello welcome to Submagic."),
            ("This is a test subtitle sentence", "This is a test subtitle sentence!"),
            ("Xin chào các bạn đã đến với kênh", "Xin chào các bạn đã đến với kênh."),
            ("Trí tuệ nhân tạo đang phát triển rất nhanh", "Trí tuệ nhân tạo đang phát triển rất nhanh!"),
        ]

        for s_prev, s_cand in merge_pairs:
            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_prev, confidence=0.90)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_cand, confidence=0.95)

            dedup.add_or_merge(seg1)
            res, merged = dedup.add_or_merge(seg2)

            self.assertTrue(
                merged,
                f"Normal Levenshtein merge failed for non-CJK pair: '{s_prev}' vs '{s_cand}'",
            )
            self.assertEqual(len(dedup.segments), 1)

        # Distinct non-CJK strings must NOT merge
        diff_pairs = [
            ("Hello welcome to Submagic", "Goodbye see you tomorrow"),
            ("Xin chào các bạn", "Hẹn gặp lại các bạn hôm nay"),
        ]
        for s_prev, s_cand in diff_pairs:
            dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)
            seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text=s_prev)
            seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text=s_cand)

            dedup.add_or_merge(seg1)
            _, merged = dedup.add_or_merge(seg2)

            self.assertFalse(merged, f"Distinct non-CJK strings falsely merged: '{s_prev}' vs '{s_cand}'")
            self.assertEqual(len(dedup.segments), 2)


class TestFeature16MaskVisualDifferenceGuard(unittest.TestCase):
    """Empirical challenge tests for Feature 16 (Mask Visual Difference Guard)."""

    def test_identical_text_with_position_jump_blocks_merge(self):
        """
        Adversarial Test: Identical subtitle text where subtitle position jumps
        significantly across frames (normalized XOR mask diff >= 0.25).
        Merge MUST be blocked!
        """
        # Case A: Top subtitle to Bottom subtitle jump
        mask_top = np.zeros((100, 400), dtype=np.uint8)
        mask_top[10:35, 50:350] = 255

        mask_bottom = np.zeros((100, 400), dtype=np.uint8)
        mask_bottom[65:90, 50:350] = 255

        diff_vertical = compute_mask_visual_diff(mask_top, mask_bottom)
        self.assertGreaterEqual(
            diff_vertical,
            0.25,
            f"Vertical position jump diff should be >= 0.25, got {diff_vertical}",
        )

        dedup_v = FuzzySubtitleDeduplicator(mask_visual_diff_threshold=0.25)
        seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="日月神教大教主")
        seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="日月神教大教主")

        dedup_v.add_or_merge(seg1, mask=mask_top)
        _, merged_v = dedup_v.add_or_merge(seg2, mask=mask_bottom)

        self.assertFalse(
            merged_v,
            "REGRESSION: Identical text with vertical position jump (diff >= 0.25) was falsely merged!",
        )
        self.assertEqual(len(dedup_v.segments), 2)

        # Case B: Left subtitle to Right subtitle horizontal jump
        mask_left = np.zeros((100, 500), dtype=np.uint8)
        mask_left[30:60, 20:200] = 255

        mask_right = np.zeros((100, 500), dtype=np.uint8)
        mask_right[30:60, 300:480] = 255

        diff_horizontal = compute_mask_visual_diff(mask_left, mask_right)
        self.assertGreaterEqual(
            diff_horizontal,
            0.25,
            f"Horizontal position jump diff should be >= 0.25, got {diff_horizontal}",
        )

        dedup_h = FuzzySubtitleDeduplicator(mask_visual_diff_threshold=0.25)
        seg3 = SubtitleSegment(id="s3", project_id="p", index=1, start_time=1.0, end_time=2.0, text="Same Speaker")
        seg4 = SubtitleSegment(id="s4", project_id="p", index=2, start_time=2.1, end_time=3.0, text="Same Speaker")

        dedup_h.add_or_merge(seg3, mask=mask_left)
        _, merged_h = dedup_h.add_or_merge(seg4, mask=mask_right)

        self.assertFalse(
            merged_h,
            "REGRESSION: Identical text with horizontal position jump (diff >= 0.25) was falsely merged!",
        )
        self.assertEqual(len(dedup_h.segments), 2)

    def test_identical_text_with_slight_frame_jitter_allows_merge(self):
        """
        Adversarial Test: Identical subtitle text with slight frame jitter
        (mask diff < 0.25). Normal merge MUST be allowed!
        """
        mask_base = np.zeros((100, 400), dtype=np.uint8)
        mask_base[20:50, 50:350] = 255

        # 1-pixel shift (camera vibration / telecine jitter)
        mask_jitter = np.zeros((100, 400), dtype=np.uint8)
        mask_jitter[21:50, 50:350] = 255

        diff = compute_mask_visual_diff(mask_base, mask_jitter)
        self.assertLess(
            diff,
            0.25,
            f"1-pixel jitter diff should be < 0.25, got {diff}",
        )

        dedup = FuzzySubtitleDeduplicator(mask_visual_diff_threshold=0.25)
        seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="日月神教大教主")
        seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="日月神教大教主")

        dedup.add_or_merge(seg1, mask=mask_base)
        res, merged = dedup.add_or_merge(seg2, mask=mask_jitter)

        self.assertTrue(
            merged,
            f"REGRESSION: Identical text with slight jitter (diff={diff:.4f} < 0.25) was not merged!",
        )
        self.assertEqual(len(dedup.segments), 1)
        self.assertEqual(res.end_time, 3.0)

    def test_mask_guard_o1_memory_and_none_handling(self):
        """
        Verify that FuzzySubtitleDeduplicator stores only _last_mask in O(1) memory
        and handles mask=None gracefully without exceptions.
        """
        dedup = FuzzySubtitleDeduplicator()
        self.assertIsNone(dedup._last_mask)

        mask1 = np.ones((50, 200), dtype=np.uint8) * 255
        seg1 = SubtitleSegment(id="s1", project_id="p", index=1, start_time=1.0, end_time=2.0, text="Hello World")
        dedup.add_or_merge(seg1, mask=mask1)
        self.assertIs(dedup._last_mask, mask1)

        # None mask must not throw and must replace _last_mask
        seg2 = SubtitleSegment(id="s2", project_id="p", index=2, start_time=2.1, end_time=3.0, text="Another Line")
        dedup.add_or_merge(seg2, mask=None)
        self.assertIsNone(dedup._last_mask)
        self.assertEqual(len(dedup.segments), 2)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
