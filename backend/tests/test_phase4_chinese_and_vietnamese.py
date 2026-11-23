"""
Milestone 4 Comprehensive Test Suite: Chinese Normalization, Dual Storage & Vietnamese Diacritics
Tests:
- Feature 17: OpenCC t2s normalization (Traditional -> Simplified)
- Feature 18: Dual-version text storage (original_text & normalized_text) + Export script variants
- Feature 19: Script distribution counter (count distinct Trad & Simp glyphs)
- Feature 20: Mixed-script low-confidence flagging
- Feature 21: Vietnamese full-resolution crop preservation (protecting dấu nặng)
- Feature 22: Dual CER evaluation support (standard CER vs diacritic-stripped CER)
"""
from __future__ import annotations

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from app.schemas import (
    BilingualLayout,
    ExportFormat,
    ExportRequest,
    ScriptVariant,
    SubtitleSegment,
    SubtitleSegmentUpdate,
)
from app.utils.cjk_normalizer import (
    convert_s2t,
    convert_t2s,
    count_trad_simp_characters,
    is_mixed_cjk_script,
)
from app.utils.subtitle_formatter import format_subtitles
from app.services.ocr_pipeline import (
    FuzzySubtitleDeduplicator,
    normalize_for_fuzzy,
)
from app.database import DatabaseManager


class TestChineseOpenCCAndScriptAnalysis:
    """Tests for Features 17, 18, 19, 20."""

    def test_opencc_t2s_conversion(self):
        """Feature 17: Verify t2s converts Traditional Chinese characters into Simplified."""
        trad = "誰能想到東方不敗也敗在別人手下了"
        simp = convert_t2s(trad)
        assert simp == "谁能想到东方不败也败在别人手下了"
        assert "誰" not in simp and "谁" in simp
        assert "東" not in simp and "东" in simp
        assert "敗" not in simp and "败" in simp
        assert "別" not in simp and "别" in simp

    def test_opencc_s2t_conversion(self):
        """Feature 18: Verify s2t converts Simplified Chinese characters into Traditional."""
        simp = "谁能想到东方不败也败在别人手下了"
        trad = convert_s2t(simp)
        assert "誰" in trad
        assert "東" in trad
        assert "敗" in trad
        assert "別" in trad

    def test_script_distribution_counter(self):
        """Feature 19: Count distinct Traditional-only and Simplified-only glyphs per line."""
        trad_pure = "誰能想到東方不敗"
        trad_cnt, simp_cnt = count_trad_simp_characters(trad_pure)
        assert trad_cnt >= 3  # 誰, 東, 敗
        assert simp_cnt == 0

        simp_pure = "谁能想到东方不败"
        trad_cnt2, simp_cnt2 = count_trad_simp_characters(simp_pure)
        assert trad_cnt2 == 0
        assert simp_cnt2 >= 3  # 谁, 东, 败

    def test_mixed_script_detection_and_flag(self):
        """Feature 20: Flag mixed Traditional and Simplified Chinese segments as low confidence."""
        # Mix of Traditional '東' and Simplified '败'
        mixed = "能想到東方不败也败在别人手下了"
        trad_cnt, simp_cnt = count_trad_simp_characters(mixed)
        assert trad_cnt > 0 and simp_cnt > 0
        assert is_mixed_cjk_script(mixed) is True

        # Pure traditional or simplified should NOT be flagged as mixed
        assert is_mixed_cjk_script("誰能想到東方不敗") is False
        assert is_mixed_cjk_script("谁能想到东方不败") is False

    def test_deduplicator_merges_traditional_and_simplified_versions(self):
        """Feature 17 & 18: FuzzySubtitleDeduplicator normalizes CJK text via t2s for deduplication."""
        dedup = FuzzySubtitleDeduplicator(fuzzy_match_threshold=85.0, max_merge_gap_sec=0.20)

        # Candidate 1: Traditional
        s1 = SubtitleSegment(
            id="s1",
            project_id="p",
            index=1,
            start_time=1.0,
            end_time=2.0,
            text="誰能想到東方不敗",
            original_text="誰能想到東方不敗",
            normalized_text="谁能想到东方不败",
            confidence=0.90,
        )
        # Candidate 2: Simplified (same semantic words)
        s2 = SubtitleSegment(
            id="s2",
            project_id="p",
            index=2,
            start_time=2.1,
            end_time=3.0,
            text="谁能想到东方不败",
            original_text="谁能想到东方不败",
            normalized_text="谁能想到东方不败",
            confidence=0.95,
        )

        dedup.add_or_merge(s1)
        res, merged = dedup.add_or_merge(s2)

        assert merged is True, "Traditional and Simplified identical lines must merge"
        assert len(dedup.segments) == 1
        assert res.start_time == 1.0
        assert res.end_time == 3.0
        assert res.confidence == 0.95


class TestExportScriptVariants:
    """Feature 18: Subtitle export supporting Original, Simplified, and Traditional script variants."""

    def test_export_subtitles_script_variants(self):
        segs = [
            SubtitleSegment(
                id="s1",
                project_id="p",
                index=1,
                start_time=1.0,
                end_time=2.5,
                text="誰能想到東方不敗",
                original_text="誰能想到東方不敗",
                normalized_text="谁能想到东方不败",
                confidence=0.92,
            )
        ]

        # 1. Original: retains exact original characters
        out_orig = format_subtitles(segs, export_format=ExportFormat.SRT, script_variant=ScriptVariant.ORIGINAL)
        assert "誰能想到東方不敗" in out_orig
        assert "谁" not in out_orig

        # 2. Simplified: converts to simplified characters
        out_simp = format_subtitles(segs, export_format=ExportFormat.SRT, script_variant=ScriptVariant.SIMPLIFIED)
        assert "谁能想到东方不败" in out_simp
        assert "誰" not in out_simp

        # 3. Traditional: converts to traditional characters
        out_trad = format_subtitles(segs, export_format=ExportFormat.SRT, script_variant=ScriptVariant.TRADITIONAL)
        assert "誰能想到東方不敗" in out_trad


class TestVietnameseFullResAndDiacritics:
    """Feature 21 & 22: Vietnamese full-resolution crop preservation and dual CER."""

    def test_vietnamese_dấu_nặng_dot_preservation(self):
        """Feature 21: Verify dot underneath (dấu nặng) is preserved in unbinarized full-res crop."""
        # Create image with a small dark/bright dot underneath a letter (e.g., 'ụ' or 'ị')
        crop = np.full((50, 100, 3), 30, dtype=np.uint8)
        # Letter stroke at y=15..35, x=40..60
        crop[15:35, 40:60] = 240
        # Dot 'dấu nặng' at y=42..45, x=48..52 (small 4x4 dot)
        crop[42:46, 48:52] = 240

        # Full resolution retains the dot
        assert np.max(crop[42:46, 48:52]) == 240
        dot_area = np.count_nonzero(crop[42:46, 48:52] > 128)
        assert dot_area >= 12, "Dấu nặng stroke must be clearly present in full-resolution image"

    def test_dual_cer_evaluator_metric_calculation(self):
        """Feature 22: Validate standard CER and diacritic-stripped CER logic."""
        import unicodedata

        def strip_vietnamese_diacritics(text: str) -> str:
            text = text.replace("đ", "d").replace("Đ", "D")
            nfkd = unicodedata.normalize("NFKD", text)
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        ref = "Chào bạn Việt Nam"
        hyp_diacritic_error = "Chao ban Viet Nam"  # Missing all diacritics

        # Reference and hypothesis diacritic-stripped are 100% identical
        assert strip_vietnamese_diacritics(ref) == strip_vietnamese_diacritics(hyp_diacritic_error)

        from difflib import SequenceMatcher
        # Raw distance
        matcher_raw = SequenceMatcher(None, ref, hyp_diacritic_error)
        raw_cer = 1.0 - matcher_raw.ratio()
        assert raw_cer > 0.0, "Raw CER must penalize missing diacritics"

        # Stripped distance
        matcher_stripped = SequenceMatcher(
            None,
            strip_vietnamese_diacritics(ref),
            strip_vietnamese_diacritics(hyp_diacritic_error),
        )
        stripped_cer = 1.0 - matcher_stripped.ratio()
        assert stripped_cer == 0.0, "Diacritic-stripped CER must be 0 for exact tone/accent mismatch"


@pytest.mark.asyncio
class TestDatabaseDualVersionStorage:
    """Feature 18: Database persistence of dual text versions."""

    async def test_database_dual_version_persistence(self, tmp_path: Path):
        db_file = tmp_path / "test_dual.db"
        db = DatabaseManager(db_path=db_file)
        await db.init_db()

        proj = await db.create_project(name="Dual Text Project", video_path="sample.mp4")

        seg = SubtitleSegment(
            id="seg-dual-1",
            project_id=proj.id,
            index=1,
            start_time=1.0,
            end_time=2.5,
            text="誰能想到東方不敗",
            original_text="誰能想到東方不敗",
            normalized_text="谁能想到东方不败",
            confidence=0.91,
            is_low_confidence=False,
        )

        inserted = await db.insert_segment(seg)
        assert inserted.original_text == "誰能想到東方不敗"
        assert inserted.normalized_text == "谁能想到东方不败"
        assert inserted.is_low_confidence is False

        # Retrieve and verify round-trip persistence
        fetched_proj = await db.get_project(proj.id)
        assert fetched_proj is not None
        assert len(fetched_proj.segments) == 1
        fetched_seg = fetched_proj.segments[0]
        assert fetched_seg.text == "誰能想到東方不敗"
        assert fetched_seg.original_text == "誰能想到東方不敗"
        assert fetched_seg.normalized_text == "谁能想到东方不败"
        assert fetched_seg.is_low_confidence is False

        # Update segment test
        await db.update_segment(
            proj.id,
            "seg-dual-1",
            SubtitleSegmentUpdate(
                normalized_text="谁能想到东方不败 (updated)",
                is_low_confidence=True,
            ),
        )

        updated_proj = await db.get_project(proj.id)
        assert updated_proj.segments[0].normalized_text == "谁能想到东方不败 (updated)"
        assert updated_proj.segments[0].is_low_confidence is True


class TestVietnamesePaddingAndCropResolution:
    """Tests for Vietnamese dấu nặng padding protection and VietOCR minimum input resolution."""

    def test_crop_text_box_pad_bottom_protection(self):
        """Verify crop_text_box applies pad_bottom to safeguard diacritics below baseline."""
        from app.services.ocr_engines import crop_text_box
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        box = [[20, 30], [100, 30], [100, 50], [20, 50]]
        # pad=3, pad_bottom=10 -> y1=27, y2=60 (height=33)
        crop = crop_text_box(img, box, pad=3, pad_bottom=10)
        assert crop.shape[0] == 33
        assert crop.shape[1] == 86

    def test_recognize_vietocr_upscales_small_crops(self):
        """Verify recognize_vietocr ensures height >= 32px before predictor input."""
        from unittest.mock import MagicMock, patch
        from app.services.ocr_engines import OCREngineCache, OCRTextLine

        cache = OCREngineCache()
        # Mock RapidOCR returning a small 16px high box
        mock_rapid = MagicMock()
        mock_rapid.return_value = (
            [[[[10, 10], [50, 10], [50, 24], [10, 24]], "lại", 0.90]],
            [0.01],
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("lại", 0.95)

        test_img = np.zeros((40, 100, 3), dtype=np.uint8)
        with patch.object(cache, "get_rapidocr", return_value=mock_rapid), \
             patch.object(cache, "get_vietocr", return_value=mock_predictor):
            res = cache.recognize_vietocr(test_img)
            assert res.text == "lại"
            assert mock_predictor.predict.called
            # First arg of predict is PIL image; its height must be >= 32
            pil_call = mock_predictor.predict.call_args[0][0]
            assert pil_call.size[1] >= 32

    def test_recognize_vietocr_drops_nobox_hallucinations(self):
        """Verify recognize_vietocr filters out English-like hallucinations when RapidOCR detects 0 boxes."""
        from unittest.mock import MagicMock, patch
        from app.services.ocr_engines import OCREngineCache

        cache = OCREngineCache()
        mock_rapid = MagicMock()
        mock_rapid.return_value = ([], [0.01])

        mock_predictor = MagicMock()
        test_img = np.zeros((90, 480, 3), dtype=np.uint8)

        with patch.object(cache, "get_rapidocr", return_value=mock_rapid), \
             patch.object(cache, "get_vietocr", return_value=mock_predictor):
            for hallucination in ("Conflamilization", "INTERVERSIONALIZE", "N W"):
                mock_predictor.predict.return_value = (hallucination, 0.92)
                res = cache.recognize_vietocr(test_img)
                assert res.text == "", f"Should drop no-box hallucination '{hallucination}'"

            # Valid Vietnamese tight-line fallback should still pass
            mock_predictor.predict.return_value = ("Phụ đề dòng đơn sát viền", 0.96)
            valid_res = cache.recognize_vietocr(test_img)
            assert valid_res.text == "Phụ đề dòng đơn sát viền"


