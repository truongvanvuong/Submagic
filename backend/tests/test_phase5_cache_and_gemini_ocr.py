"""
Milestone 5 Comprehensive Test Suite: Translation Cache 5-Tuple Key & Selective Gemini Vision OCR
Tests:
- Feature 23: Translation cache composite 5-tuple key (source_text, target_lang, context_type, model_name, prompt_version)
- Feature 24: Translation cache clear endpoint & programmatic purging
- Feature 25: Selective Gemini OCR fallback exclusively for low-confidence segments
- Feature 26: Pre-activation Gemini OCR token and cost estimator
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.database import DatabaseManager
from app.schemas import (
    GeminiOCREstimateRequest,
    GeminiOCRExecuteRequest,
    SubtitleSegment,
    SubtitleSegmentUpdate,
)
from app.services.ocr_engines import OCRFrameResult
from app.routers.jobs import (
    clear_translation_cache_endpoint,
    estimate_gemini_ocr_endpoint,
    execute_gemini_ocr_fallback,
)
from app.config import settings


@pytest.mark.asyncio
class TestTranslationCache5TupleCompositeKey:
    """Feature 23 & 24: Composite 5-tuple key and cache clear functionality."""

    async def test_cache_5tuple_distinct_models_and_prompt_versions(self, tmp_path: Path):
        db_file = tmp_path / "test_cache5.db"
        db = DatabaseManager(db_path=db_file)
        await db.init_db()

        src = "天下風雲出我輩"
        lang = "vi"
        ctx = "historical"

        # Model A with prompt v1.0
        entry_a = (src, lang, ctx, "gemini-2.5-flash", "v1.0", "Thiên hạ phong vân xuất ngã bối (v1)")
        # Model A with prompt v2.0
        entry_b = (src, lang, ctx, "gemini-2.5-flash", "v2.0", "Thiên hạ phong vân xuất ngã bối (v2)")
        # Model B with prompt v1.0
        entry_c = (src, lang, ctx, "gemini-3.8-flash", "v1.0", "Thiên hạ phong vân xuất ngã bối (Gemini 3.8)")

        await db.save_cached_translations([entry_a, entry_b, entry_c])

        # Query specific model and prompt version
        res_a = await db.get_cached_translations(lang, ctx, [src], model_name="gemini-2.5-flash", prompt_version="v1.0")
        assert res_a.get(src) == "Thiên hạ phong vân xuất ngã bối (v1)"

        res_b = await db.get_cached_translations(lang, ctx, [src], model_name="gemini-2.5-flash", prompt_version="v2.0")
        assert res_b.get(src) == "Thiên hạ phong vân xuất ngã bối (v2)"

        res_c = await db.get_cached_translations(lang, ctx, [src], model_name="gemini-3.8-flash", prompt_version="v1.0")
        assert res_c.get(src) == "Thiên hạ phong vân xuất ngã bối (Gemini 3.8)"

    async def test_clear_translation_cache_all_and_filtered(self, tmp_path: Path):
        db_file = tmp_path / "test_clear.db"
        db = DatabaseManager(db_path=db_file)
        await db.init_db()

        entries = [
            ("Hello", "vi", "casual", "default", "v1.0", "Xin chào"),
            ("Goodbye", "vi", "casual", "default", "v1.0", "Tạm biệt"),
            ("Good morning", "en", "casual", "default", "v1.0", "Good morning"),
        ]
        await db.save_cached_translations(entries)

        # Filtered purge by target_lang
        del_count = await db.clear_translation_cache(target_lang="en")
        assert del_count == 1

        # Check remaining
        cached_vi = await db.get_cached_translations("vi", "casual", ["Hello", "Goodbye"])
        assert len(cached_vi) == 2

        # Purge all remaining
        del_all = await db.clear_translation_cache()
        assert del_all == 2

        cached_empty = await db.get_cached_translations("vi", "casual", ["Hello"])
        assert len(cached_empty) == 0


@pytest.mark.asyncio
class TestSelectiveGeminiVisionOCRFallback:
    """Feature 25 & 26: Pre-activation cost estimation and low-confidence routing."""

    async def test_gemini_ocr_cost_estimator(self, tmp_path: Path):
        db_file = tmp_path / "test_est.db"
        test_db = DatabaseManager(db_path=db_file)
        await test_db.init_db()

        with patch("app.routers.jobs.db", test_db):
            proj = await test_db.create_project(name="Gemini OCR Est", video_path="v.mp4")

            # Insert 5 segments: 2 high confidence, 3 low confidence
            segs = [
                SubtitleSegment(project_id=proj.id, index=1, start_time=1.0, end_time=2.0, text="Dòng rõ nét 1", confidence=0.95, is_low_confidence=False),
                SubtitleSegment(project_id=proj.id, index=2, start_time=2.1, end_time=3.0, text="Dòng mờ 1", confidence=0.45, is_low_confidence=True),
                SubtitleSegment(project_id=proj.id, index=3, start_time=3.1, end_time=4.0, text="Dòng rõ nét 2", confidence=0.92, is_low_confidence=False),
                SubtitleSegment(project_id=proj.id, index=4, start_time=4.1, end_time=5.0, text="Dòng mờ 2", confidence=0.50, is_low_confidence=True),
                SubtitleSegment(project_id=proj.id, index=5, start_time=5.1, end_time=6.0, text="Dòng mờ 3", confidence=0.40, is_low_confidence=True),
            ]
            for s in segs:
                await test_db.insert_segment(s)

            # 1. Request with max_lines = 2
            req_capped = GeminiOCREstimateRequest(project_id=proj.id, max_lines=2)
            est_capped = await estimate_gemini_ocr_endpoint(req_capped)

            assert est_capped.total_segments == 5
            assert est_capped.low_confidence_segments == 3
            assert est_capped.lines_to_process == 2
            assert est_capped.estimated_prompt_tokens == 2 * 280
            assert est_capped.estimated_cost_usd > 0.0

            # 2. Request with max_lines = 10 (uncapped)
            req_all = GeminiOCREstimateRequest(project_id=proj.id, max_lines=10)
            est_all = await estimate_gemini_ocr_endpoint(req_all)
            assert est_all.lines_to_process == 3

    async def test_gemini_ocr_fallback_execution_on_low_confidence_segments(self, tmp_path: Path):
        db_file = tmp_path / "test_exec.db"
        test_db = DatabaseManager(db_path=db_file)
        await test_db.init_db()

        with patch("app.routers.jobs.db", test_db):
            proj = await test_db.create_project(name="Gemini Fallback Exec", video_path="v.mp4")

            # Create dummy crop image
            crops_dir = tmp_path / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            crop_file = crops_dir / f"{proj.id}_0_1.000.jpg"
            dummy_img = np.zeros((40, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(crop_file), dummy_img)

            seg_low = SubtitleSegment(
                id="seg-low-1",
                project_id=proj.id,
                index=1,
                start_time=1.0,
                end_time=2.5,
                text="Chữ bị mờ khó đọc",
                confidence=0.45,
                is_low_confidence=True,
                crop_image_url=f"/media/{crop_file.name}",
            )
            seg_high = SubtitleSegment(
                id="seg-high-2",
                project_id=proj.id,
                index=2,
                start_time=3.0,
                end_time=4.5,
                text="Chữ chuẩn nét",
                confidence=0.98,
                is_low_confidence=False,
            )
            await test_db.insert_segment(seg_low)
            await test_db.insert_segment(seg_high)

            # Mock Gemini recognition
            mock_gemini_result = OCRFrameResult(
                text="Chữ đã được Gemini sửa đúng",
                confidence=0.97,
                lines=[],
                engine_used="gemini",
            )

            with patch("app.routers.jobs.settings.crops_dir", crops_dir), \
                 patch("app.services.ocr_engines.ocr_engine_cache.recognize_gemini", return_value=mock_gemini_result):

                res = await execute_gemini_ocr_fallback(
                    GeminiOCRExecuteRequest(project_id=proj.id, max_lines=5)
                )

                assert res["processed_segments"] == 1
                assert "seg-low-1" in res["updated_segment_ids"]

                # Verify database update: text is updated and is_low_confidence is cleared
                refreshed = await test_db.get_project(proj.id)
                upd_seg = next(s for s in refreshed.segments if s.id == "seg-low-1")
                assert upd_seg.text == "Chữ đã được Gemini sửa đúng"
                assert upd_seg.is_low_confidence is False
                assert upd_seg.confidence >= 0.95
