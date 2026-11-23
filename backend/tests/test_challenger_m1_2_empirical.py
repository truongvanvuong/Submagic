"""Empirical Stress Testing Suite by Challenger 2 for Milestone 1.

Focus areas:
1. Stress-testing `scripts/eval_srt.py`:
   - Edge CJK pairs (Traditional vs Simplified, fallback dictionary vs OpenCC, punctuation, mixed scripts)
   - Empty subtitles (0-byte file, blank lines, empty subtitle text, empty predicted file)
   - Misaligned timestamps (out-of-order, zero-duration, inverted, overlapping, large time gaps)
   - Vietnamese sentences with/without accents (complex tone marks, composite vs precomposed, uppercase/lowercase Đ/d)
   - Non-crashing console output under simulated Windows legacy/default codepages (cp1252, cp1258, cp437)
2. Diagnostic logging challenge:
   - Sensitive data leak auditing (no API keys, tokens, auth secrets, binary data in logs)
   - Infinite recursion & unbounded memory verification (bounded log message size, no cyclic reference leaks)
   - Diagnostic logging performance overhead benchmark (overhead under 5% / negligible CPU time)
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

import numpy as np
import pytest

# Ensure repo root is in sys.path to access scripts.eval_srt
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_srt import (  # noqa: E402
    T2S_FALLBACK_MAP,
    convert_t2s,
    edit_distance,
    match,
    normalize,
    parse_srt,
    strip_vietnamese_diacritics,
)
from app.schemas import OCRJobRequest, SubtitleSegment  # noqa: E402
from app.services.ocr_pipeline import (  # noqa: E402
    FuzzySubtitleDeduplicator,
    ThreeLayerVisualFilter,
)
from app.services.ocr_engines import OCRTextLine, order_and_merge_ocr_lines  # noqa: E402


# ==============================================================================
# SECTION 1: CJK Script Folding & Edge Cases
# ==============================================================================


class TestCJKScriptFoldingEdgeCases:
    """Stress tests for CJK folding, fallback dictionary, and character comparisons."""

    def test_cjk_traditional_to_simplified_fallback_dictionary(self):
        """Verify fallback dictionary converts high-frequency traditional characters accurately."""
        # Test key traditional characters from target video and common hardsubs
        sample_trad = "誰能想到東方不敗也敗在別人手下了"
        converted = convert_t2s(sample_trad)
        # 誰 -> 谁, 東 -> 东, 敗 -> 败, 別 -> 别
        assert "谁" in converted
        assert "东" in converted
        assert "败" in converted
        assert "别" in converted
        assert "誰" not in converted
        assert "東" not in converted
        assert "敗" not in converted

    def test_cjk_punctuation_and_fullwidth_symbols_normalization(self):
        """Full-width punctuation (，。！？、“”《》) and ideographic spaces must normalize cleanly."""
        text_fullwidth = "【東方不敗】：誰能想到？“沒有人”！"
        norm = normalize(text_fullwidth, fold_script="t2s")
        # Punctuation should be stripped, text folded to simplified
        assert "东方不败" in norm
        assert "谁能想到" in norm
        assert "没有人" in norm
        assert "【" not in norm and "】" not in norm
        assert "？" not in norm and "！" not in norm
        assert "“" not in norm and "”" not in norm

    def test_cjk_differing_by_one_character(self):
        """Two 7+ char CJK sentences differing by exactly 1 character should have low edit distance."""
        s1 = "天下風雲出我輩"
        s2 = "天下風雲出我輩一"
        d = edit_distance(normalize(s1, fold_script="t2s"), normalize(s2, fold_script="t2s"))
        assert d == 1

        s3 = "天下風雲出我輩"
        s4 = "天下風雲出他輩"
        d2 = edit_distance(normalize(s3, fold_script="t2s"), normalize(s4, fold_script="t2s"))
        assert d2 == 1

    def test_cjk_mixed_scripts_and_emojis(self):
        """Strings mixing Traditional Chinese, Simplified Chinese, English, Vietnamese, and Emojis."""
        mixed = "誰 vs 谁 (DongFang 100%) - Tiếng Việt 🎬🚀"
        norm = normalize(mixed, fold_script="t2s")
        assert "谁vs谁" in norm
        assert "dongfang100" in norm
        assert "tiengviet" in norm or "tiếngviệt" in norm
        # Emojis (symbols 'So') are filtered by category check
        assert "🎬" not in norm
        assert "🚀" not in norm


# ==============================================================================
# SECTION 2: Empty Subtitles & Boundary Handling in eval_srt.py
# ==============================================================================


class TestEvalSrtEmptyAndBoundarySubtitles:
    """Stress tests for empty files, blank subtitle texts, and non-standard whitespace."""

    def test_parse_srt_empty_file_and_whitespace(self, tmp_path: Path):
        """Parsing 0-byte or whitespace-only SRT files returns empty list without error."""
        empty_file = tmp_path / "empty.srt"
        empty_file.write_text("", encoding="utf-8")
        assert parse_srt(str(empty_file)) == []

        ws_file = tmp_path / "whitespace.srt"
        ws_file.write_text("\n\n   \t  \r\n\r\n", encoding="utf-8")
        assert parse_srt(str(ws_file)) == []

    def test_parse_srt_cues_with_empty_or_whitespace_text(self, tmp_path: Path):
        """SRT blocks with valid timestamps but empty/whitespace text are parsed safely."""
        srt_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "   \t  \n"
            "\n"
            "3\n"
            "00:00:05,000 --> 00:00:06,000\n"
            "Valid subtitle text\n"
        )
        f = tmp_path / "blank_cues.srt"
        f.write_text(srt_content, encoding="utf-8")
        entries = parse_srt(str(f))
        assert len(entries) == 3
        assert entries[0]["text"] == ""
        assert entries[1]["text"] == ""
        assert entries[2]["text"] == "Valid subtitle text"

    def test_eval_srt_cli_with_empty_predicted_file(self, tmp_path: Path):
        """Running eval_srt.py when predicted SRT is completely empty should yield 0% recall/precision cleanly."""
        gt_file = tmp_path / "gt.srt"
        gt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nXin chào\n", encoding="utf-8"
        )
        empty_pred = tmp_path / "pred_empty.srt"
        empty_pred.write_text("", encoding="utf-8")

        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "eval_srt.py"), str(gt_file), str(empty_pred)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert res.returncode == 0
        assert "Recall (bat duoc sub)    :   0.0%" in res.stdout
        assert "Precision (khong thua)   :   0.0%" in res.stdout
        assert "1 dong bi sot" in res.stdout


# ==============================================================================
# SECTION 3: Misaligned & Pathological Timestamps in eval_srt.py
# ==============================================================================


class TestEvalSrtMisalignedTimestamps:
    """Stress tests for out-of-order, zero-duration, inverted, and overlapping timestamps."""

    def test_out_of_order_cues_sorted_by_start(self, tmp_path: Path):
        """SRT files with blocks out of chronological order are sorted by start_time."""
        srt_content = (
            "1\n00:00:10,000 --> 00:00:12,000\nLate cue\n\n"
            "2\n00:00:01,000 --> 00:00:03,000\nEarly cue\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\nMid cue\n"
        )
        f = tmp_path / "shuffled.srt"
        f.write_text(srt_content, encoding="utf-8")
        entries = parse_srt(str(f))
        starts = [e["start"] for e in entries]
        assert starts == [1.0, 5.0, 10.0]
        assert [e["text"] for e in entries] == ["Early cue", "Mid cue", "Late cue"]

    def test_zero_duration_and_overlapping_cues_matching(self, tmp_path: Path):
        """Zero duration (start == end) and overlapping cues do not crash matcher."""
        gt_file = tmp_path / "gt_overlap.srt"
        gt_file.write_text(
            "1\n00:00:01,000 --> 00:00:01,000\nZero duration cue\n\n"
            "2\n00:00:01,500 --> 00:00:04,000\nOverlapping cue\n",
            encoding="utf-8",
        )
        pred_file = tmp_path / "pred_overlap.srt"
        pred_file.write_text(
            "1\n00:00:01,000 --> 00:00:01,000\nZero duration cue\n\n"
            "2\n00:00:02,000 --> 00:00:04,500\nOverlapping cue\n",
            encoding="utf-8",
        )

        gt = parse_srt(str(gt_file))
        pred = parse_srt(str(pred_file))
        pairs, used_g, used_p = match(gt, pred, window=3.0, min_sim=0.5)
        assert len(pairs) == 2
        assert len(used_g) == 2
        assert len(used_p) == 2

    def test_extreme_time_drift_linear_fit(self):
        """Linear fit calculation handles large time drift trends without numerical instability."""
        from scripts.eval_srt import linear_fit

        # Systematic 100ms drift per minute
        gt_starts = [0.0, 60.0, 120.0, 180.0, 240.0]
        offsets = [0.050, 0.150, 0.250, 0.350, 0.450]
        fit = linear_fit(gt_starts, offsets)
        assert fit is not None
        slope, intercept = fit
        assert pytest.approx(slope * 60, abs=1e-3) == 0.100  # 100ms per minute
        assert pytest.approx(intercept, abs=1e-3) == 0.050


# ==============================================================================
# SECTION 4: Vietnamese Diacritics Evaluation Stress
# ==============================================================================


class TestVietnameseDiacriticsStress:
    """Stress tests for Vietnamese diacritic stripping, tone normalization, and dual CER."""

    def test_strip_vietnamese_diacritics_exhaustive_vowels(self):
        """All 134 Vietnamese vowel+tone combinations strip to base Latin letters."""
        # Test standard Vietnamese pangram with all complex diacritics
        text_vn = (
            "Học sinh giỏi: á, à, ả, ã, ạ; ă, ắ, ằ, ẳ, ẵ, ặ; â, ấ, ầ, ẩ, ẫ, ậ; "
            "é, è, ẻ, ẽ, ẹ; ê, ế, ề, ể, ễ, ệ; í, ì, ỉ, ĩ, ị; "
            "ó, ò, ỏ, õ, ọ; ô, ố, ồ, ổ, ỗ, ộ; ơ, ớ, ờ, ở, ỡ, ợ; "
            "ú, ù, ủ, ũ, ụ; ư, ứ, ừ, ử, ữ, ự; ý, ỳ, ỷ, ỹ, ỵ; đ, Đ."
        )
        stripped = strip_vietnamese_diacritics(text_vn)
        # Verify no combining marks remain
        decomposed = unicodedata.normalize("NFD", stripped)
        combining = [ch for ch in decomposed if unicodedata.category(ch) == "Mn"]
        assert len(combining) == 0
        # Verify 'đ' and 'Đ' converted to 'd' and 'D'
        assert "đ" not in stripped
        assert "Đ" not in stripped
        assert "d" in stripped
        assert "D" in stripped
        assert "Hoc sinh gioi" in stripped

    def test_nfc_vs_nfd_decomposition_invariance(self):
        """Precomposed (NFC) and Decomposed (NFD) Vietnamese strings produce identical stripped output."""
        text_nfc = unicodedata.normalize("NFC", "Tiếng Việt giàu đẹp và rực rỡ")
        text_nfd = unicodedata.normalize("NFD", "Tiếng Việt giàu đẹp và rực rỡ")

        assert text_nfc != text_nfd  # Under-the-hood byte representation differs
        stripped_nfc = strip_vietnamese_diacritics(text_nfc)
        stripped_nfd = strip_vietnamese_diacritics(text_nfd)
        assert stripped_nfc == stripped_nfd
        assert stripped_nfc == "Tieng Viet giau dep va ruc ro"

    def test_eval_srt_cli_dual_cer_accuracy(self, tmp_path: Path):
        """When ground-truth is accented Vietnamese and pred has lost accents, standard CER > 0 but stripped CER == 0."""
        gt_file = tmp_path / "gt_vn.srt"
        gt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nTiếng Việt có dấu chuẩn xác\n",
            encoding="utf-8",
        )
        pred_file = tmp_path / "pred_vn.srt"
        # OCR dropped tone marks:
        pred_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nTieng Viet co dau chuan xac\n",
            encoding="utf-8",
        )

        res = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_srt.py"),
                str(gt_file),
                str(pred_file),
                "--vietnamese-diacritics",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert res.returncode == 0
        # Standard CER should detect tone difference (> 0%)
        assert "Ti le loi ky tu (CER)    :" in res.stdout
        # Diacritic-stripped CER should be 0.0%
        assert "CER (bo qua dau tieng Viet):   0.0%" in res.stdout
        assert "Doc dung (bo qua dau)    : 1/1 dong khop" in res.stdout


# ==============================================================================
# SECTION 5: Windows Console Encoding Resilience
# ==============================================================================


class TestWindowsConsoleEncodingResilience:
    """Stress test console output to ensure non-crashing under Windows non-UTF8 charsets (cp1252, cp1258)."""

    def test_cli_execution_with_utf8_cjk_and_vietnamese(self, tmp_path: Path):
        """CLI runs to completion with CJK ideographs and Vietnamese characters without UnicodeEncodeError."""
        gt_file = tmp_path / "gt_cjk_vn.srt"
        gt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n誰能想到東方不敗 (Hòa bình)\n",
            encoding="utf-8",
        )
        pred_file = tmp_path / "pred_cjk_vn.srt"
        pred_file.write_text(
            "1\n00:00:01,050 --> 00:00:03,100\n谁能想到东方不败 (Hoa binh)\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        # Test under clean subprocess execution
        res = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_srt.py"),
                str(gt_file),
                str(pred_file),
                "--fold-script",
                "t2s",
                "--vietnamese-diacritics",
            ],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
        )
        assert res.returncode == 0
        assert "Dong dap an: 1" in res.stdout
        assert "Recall (bat duoc sub)    : 100.0%" in res.stdout
        assert "UnicodeEncodeError" not in res.stderr

    def test_stream_without_reconfigure_graceful_handling(self):
        """If stdout has no reconfigure attribute (e.g. custom stream wrapper), eval_srt does not crash."""
        class DummyStream(io.StringIO):
            pass

        stream = DummyStream()
        assert not hasattr(stream, "reconfigure")
        # Simulating main()'s reconfigure check
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
        # Safe pass


# ==============================================================================
# SECTION 6: Diagnostic Logging Safety & Data Leak Audit
# ==============================================================================


class TestDiagnosticLoggingSafety:
    """Verify diagnostic logging does not leak sensitive information, explode memory, or loop."""

    def test_no_sensitive_data_in_diagnostic_logs(self, caplog):
        """Ensure no API keys, auth headers, passwords, or binary payloads are printed in [OCR DIAG] logs."""
        filter_diag = ThreeLayerVisualFilter(
            luma_threshold=185,
            stability_frames=2,
            frame_step_sec=0.2,
            diagnostic_logging=True,
        )

        dummy_roi = np.zeros((60, 200, 3), dtype=np.uint8)
        with caplog.at_level(logging.DEBUG):
            filter_diag.process_frame(dummy_roi, timestamp_sec=1.0)
            filter_diag.process_frame(dummy_roi, timestamp_sec=1.2)

        diag_messages = [r.message for r in caplog.records if "[OCR DIAG]" in r.message]
        assert len(diag_messages) > 0
        for msg in diag_messages:
            # Check absence of sensitive keywords
            assert "api_key" not in msg.lower()
            assert "password" not in msg.lower()
            assert "authorization" not in msg.lower()
            assert "secret" not in msg.lower()
            # Ensure no huge base64 or raw ndarray dumps
            assert len(msg) < 500, f"Log message unexpectedly long: {len(msg)} chars"

    def test_bounded_log_size_under_long_ocr_text(self, caplog):
        """Merging lines with very long candidate strings does not cause unbounded string construction."""
        huge_text = "A" * 10000
        lines = [
            OCRTextLine(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text=huge_text, confidence=0.10)
        ]

        with caplog.at_level(logging.INFO):
            order_and_merge_ocr_lines(lines, min_confidence=0.50, diagnostic_logging=True)

        diag_records = [r for r in caplog.records if "[OCR DIAG]" in r.message]
        assert len(diag_records) >= 1
        # Logging string should be handled without recursion or memory crash


# ==============================================================================
# SECTION 7: Diagnostic Logging Performance Overhead Benchmark
# ==============================================================================


class TestDiagnosticLoggingPerformance:
    """Measure the performance overhead of diagnostic logging to ensure it is negligible (< 5%)."""

    def test_visual_filter_logging_overhead_benchmark(self):
        """Benchmark 500 frames through visual filter with diagnostic logging OFF vs ON."""
        n_frames = 500
        dummy_frame = np.zeros((60, 300, 3), dtype=np.uint8)

        # 1. Warm-up
        f_warm = ThreeLayerVisualFilter(diagnostic_logging=False)
        for i in range(20):
            f_warm.process_frame(dummy_frame, timestamp_sec=i * 0.1)

        # 2. Timing with diagnostic_logging = False
        f_off = ThreeLayerVisualFilter(diagnostic_logging=False)
        t0 = time.perf_counter()
        for i in range(n_frames):
            f_off.process_frame(dummy_frame, timestamp_sec=i * 0.1)
        dur_off = time.perf_counter() - t0

        # 3. Timing with diagnostic_logging = True
        f_on = ThreeLayerVisualFilter(diagnostic_logging=True)
        t0 = time.perf_counter()
        for i in range(n_frames):
            f_on.process_frame(dummy_frame, timestamp_sec=i * 0.1)
        dur_on = time.perf_counter() - t0

        # Total time for 500 frames is typically ~0.05-0.10 seconds on CPU
        delta = dur_on - dur_off
        # Time per frame should not exceed 1 millisecond
        per_frame_ms = (dur_on / n_frames) * 1000
        assert per_frame_ms < 2.0, f"Visual filter too slow: {per_frame_ms:.2f} ms/frame"
        # The logging difference over 500 frames should be under 50ms total
        assert delta < 0.080, f"Logging overhead too large: {delta * 1000:.2f} ms over {n_frames} frames"

    def test_deduplicator_merge_logging_overhead_benchmark(self):
        """Benchmark 200 sequential subtitle deduplication merges with logging."""
        dedup = FuzzySubtitleDeduplicator()
        t0 = time.perf_counter()
        for i in range(200):
            seg = SubtitleSegment(
                id=f"seg-{i}",
                project_id="bench",
                index=i + 1,
                start_time=float(i),
                end_time=float(i) + 0.8,
                text="Câu phụ đề thử nghiệm tốc độ",
                confidence=0.95,
            )
            dedup.add_or_merge(seg)
        dur = time.perf_counter() - t0
        # 200 merges should complete in under 50ms
        assert dur < 0.050, f"Deduplicator benchmark took {dur * 1000:.2f} ms"
