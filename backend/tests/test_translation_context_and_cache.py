"""
Unit and Integration Tests for Subtitle Translation:
- Translation contexts (Cổ trang, Hiện đại, Đời thường)
- Granular retry on missing/extra rows in batch
- Persistent SQLite translation caching & incremental translation
- Trailing period removal on translated lines
- Pre-translation token estimation endpoint
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import (
    LLMEstimateRequest,
    LLMJobRequest,
    SubtitleSegment,
    TranslationContext,
    WebSocketEvent,
)
from app.services import llm_service
from app.services.llm_service import (
    _call_gemini_batch_postprocess,
    _get_context_instruction,
    estimate_llm_cost,
    run_llm_postprocess_pipeline,
)
from app.utils.subtitle_formatter import remove_trailing_period


@pytest.fixture(autouse=True)
async def init_test_db(tmp_path: Path):
    db.db_path = tmp_path / "test_translation.sqlite3"
    await db.init_db()


def test_translation_context_prompts():
    """Verify that each translation context provides distinct tone and pronoun guidelines."""
    hist = _get_context_instruction(TranslationContext.HISTORICAL)
    assert "CỔ TRANG" in hist
    assert "ta/ngươi" in hist
    assert "Hán-Việt" in hist

    mod = _get_context_instruction(TranslationContext.MODERN)
    assert "HIỆN ĐẠI" in mod
    assert "tôi/bạn" in mod
    assert "gãy gọn" in mod

    cas = _get_context_instruction(TranslationContext.CASUAL)
    assert "ĐỜI THƯỜNG" in cas
    assert "khẩu ngữ" in cas


@pytest.mark.asyncio
async def test_batch_missing_rows_granular_retry(tmp_path: Path):
    """When a batch returns fewer rows than requested, verify only the missing rows are retried."""
    proj = await db.create_project(name="Retry Proj", video_path="dummy.mp4")
    seg1 = SubtitleSegment(project_id=proj.id, index=1, start_time=0.0, end_time=2.0, text="Xin chào đại hiệp")
    seg2 = SubtitleSegment(project_id=proj.id, index=2, start_time=2.1, end_time=4.0, text="Tại hạ xin cáo từ")
    seg3 = SubtitleSegment(project_id=proj.id, index=3, start_time=4.1, end_time=6.0, text="Bảo trọng")
    await db.insert_segment(seg1)
    await db.insert_segment(seg2)
    await db.insert_segment(seg3)

    call_count = 0
    called_batches: list[list[str]] = []

    def mock_call_batch(batch, fix_ocr_spelling, translate_to, api_key, **kwargs):
        nonlocal call_count
        call_count += 1
        called_batches.append([s.text for s in batch])
        if call_count == 1:
            # First call returns only seg1 and seg3 (omitting seg2)
            return [
                {"id": batch[0].id, "tr": "Greetings hero."},
                {"id": batch[2].id, "tr": "Take care."},
            ]
        else:
            # Retry call returns the missing seg2
            return [
                {"id": batch[0].id, "tr": "I take my leave."},
            ]

    req = LLMJobRequest(
        project_id=proj.id,
        gemini_api_key="test-key",
        translate_to="en",
        translation_context=TranslationContext.HISTORICAL,
        force_retranslate=True,
    )

    async def noop_emit(ev):
        pass

    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_call_batch):
        res = await run_llm_postprocess_pipeline(req, job_id="job-retry", emit=noop_emit)

    assert call_count == 2
    # Verify the first call received all 3 segments
    assert len(called_batches[0]) == 3
    # Verify the retry call ONLY received the 1 missing segment ("Tại hạ xin cáo từ")!
    assert len(called_batches[1]) == 1
    assert called_batches[1][0] == "Tại hạ xin cáo từ"

    # Verify all 3 segments ended up properly translated and trailing period stripped
    assert len(res) == 3
    assert res[0].translated_text == "Greetings hero"
    assert res[1].translated_text == "I take my leave"
    assert res[2].translated_text == "Take care"


@pytest.mark.asyncio
async def test_persistent_cache_hit_and_skip_api_call(tmp_path: Path):
    """Verify that cached translations are reused without triggering extra LLM API calls."""
    proj = await db.create_project(name="Cache Test Proj", video_path="dummy.mp4")
    seg1 = SubtitleSegment(project_id=proj.id, index=1, start_time=1.0, end_time=2.5, text="Xin chào bạn hiền")
    seg2 = SubtitleSegment(project_id=proj.id, index=2, start_time=3.0, end_time=4.5, text="Hôm nay trời đẹp quá")
    await db.insert_segment(seg1)
    await db.insert_segment(seg2)

    api_call_count = 0

    def mock_call_batch(batch, fix_ocr_spelling, translate_to, api_key, **kwargs):
        nonlocal api_call_count
        api_call_count += 1
        return [
            {"id": s.id, "tr": f"Translated [{s.text}]."}
            for s in batch
        ]

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    req = LLMJobRequest(
        project_id=proj.id,
        gemini_api_key="test-key",
        translate_to="en",
        translation_context=TranslationContext.CASUAL,
    )

    # 1. First run: cache is empty, calls API
    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_call_batch):
        res1 = await run_llm_postprocess_pipeline(req, job_id="job-1", emit=collect_event)

    assert api_call_count == 1
    # Both lines received translations and trailing period was stripped
    assert res1[0].translated_text == "Translated [Xin chào bạn hiền]"
    assert res1[1].translated_text == "Translated [Hôm nay trời đẹp quá]"

    # Verify translations were saved to translation_cache in database
    cached = await db.get_cached_translations(
        target_lang="en",
        context_type="casual",
        source_texts=["Xin chào bạn hiền", "Hôm nay trời đẹp quá"],
    )
    assert len(cached) == 2
    assert "Translated [Xin chào bạn hiền]" in cached["Xin chào bạn hiền"]

    # 2. Second run on another project with same sentences: should hit cache directly, api_call_count unchanged
    proj2 = await db.create_project(name="Cache Test Proj 2", video_path="dummy2.mp4")
    seg2_1 = SubtitleSegment(project_id=proj2.id, index=1, start_time=1.0, end_time=2.5, text="Xin chào bạn hiền")
    await db.insert_segment(seg2_1)

    req2 = LLMJobRequest(
        project_id=proj2.id,
        gemini_api_key="test-key",
        translate_to="en",
        translation_context=TranslationContext.CASUAL,
    )

    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_call_batch):
        res2 = await run_llm_postprocess_pipeline(req2, job_id="job-2", emit=collect_event)

    assert api_call_count == 1  # No new API calls!
    assert res2[0].translated_text == "Translated [Xin chào bạn hiền]"


@pytest.mark.asyncio
async def test_incremental_translation_only_sends_modified_lines(tmp_path: Path):
    """When some lines are already translated, only un-translated or modified lines are sent."""
    proj = await db.create_project(name="Incremental Proj", video_path="dummy.mp4")
    # Seg 1 has translation already
    seg1 = SubtitleSegment(
        project_id=proj.id, index=1, start_time=1.0, end_time=2.0,
        text="Dòng cũ", translated_text="Old translation"
    )
    # Seg 2 is new/untranslated
    seg2 = SubtitleSegment(
        project_id=proj.id, index=2, start_time=2.5, end_time=3.5,
        text="Dòng mới cần dịch", translated_text=None
    )
    await db.insert_segment(seg1)
    await db.insert_segment(seg2)

    called_batches: list[list[SubtitleSegment]] = []

    def mock_call_batch(batch, fix_ocr_spelling, translate_to, api_key, **kwargs):
        called_batches.append(list(batch))
        return [{"id": s.id, "tr": f"Trans {s.text}"} for s in batch]

    req = LLMJobRequest(
        project_id=proj.id,
        gemini_api_key="key",
        translate_to="en",
        force_retranslate=False,
    )

    async def noop_emit(ev):
        pass

    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_call_batch):
        updated = await run_llm_postprocess_pipeline(req, job_id="job-inc", emit=noop_emit)

    # Only 1 segment (seg2) was sent to the API
    assert len(called_batches) == 1
    assert len(called_batches[0]) == 1
    assert called_batches[0][0].id == seg2.id
    assert updated[0].translated_text == "Old translation"
    assert updated[1].translated_text == "Trans Dòng mới cần dịch"


def test_trailing_period_removal_in_translation():
    """Verify that periods at sentence end are stripped while keeping ?, !, ..., numbers and abbreviations."""
    assert remove_trailing_period("Hello world.") == "Hello world"
    assert remove_trailing_period("Are you sure?") == "Are you sure?"
    assert remove_trailing_period("Watch out!") == "Watch out!"
    assert remove_trailing_period("Wait for me...") == "Wait for me..."
    assert remove_trailing_period("Version 3.5") == "Version 3.5"
    assert remove_trailing_period("Contact Mr. Smith") == "Contact Mr. Smith"


@pytest.mark.asyncio
async def test_llm_estimation_api(tmp_path: Path):
    """Test estimate_llm_cost calculation taking into account cached segments and word counts."""
    proj = await db.create_project(name="Estimate Proj", video_path="dummy.mp4")
    # Insert 10 segments of ~5 words each
    for i in range(1, 11):
        await db.insert_segment(
            SubtitleSegment(
                project_id=proj.id,
                index=i,
                start_time=float(i),
                end_time=float(i) + 1.0,
                text=f"Đây là câu phụ đề số {i}",
            )
        )

    client = TestClient(app)
    res = client.post(
        "/api/jobs/estimate-llm",
        json={
            "project_id": proj.id,
            "translate_to": "en",
            "translation_context": "casual",
            "force_retranslate": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_segments"] == 10
    assert data["cached_segments"] == 0
    assert data["segments_to_translate"] == 10
    assert data["estimated_total_tokens"] > 0
    assert data["estimated_cost_usd"] >= 0.0
