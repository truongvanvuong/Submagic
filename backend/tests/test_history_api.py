"""Tests for Extraction History persistence and REST API."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.database import db
from app.main import app


@pytest.mark.asyncio
async def test_extraction_history_lifecycle():
    await db.init_db()
    proj = await db.create_project(name="Test History Project", video_path="test.mp4")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Add a history record directly via db
        hist = await db.add_history_record(
            project_id=proj.id,
            mode="stt",
            language="auto",
            detected_language="vi",
            segment_count=5,
            export_srt_path=None,
            duration_sec=12.5,
            elapsed_sec=3.2,
        )
        assert hist["id"] is not None
        assert hist["mode"] == "stt"
        assert hist["detected_language"] == "vi"

        # 2. Query history list via API
        resp = await client.get("/api/history")
        assert resp.status_code == 200
        records = resp.json()
        assert len(records) >= 1
        found = next((r for r in records if r["id"] == hist["id"]), None)
        assert found is not None
        assert found["project_id"] == proj.id
        assert found["segment_count"] == 5

        # 3. Delete history item via API
        del_resp = await client.delete(f"/api/history/{hist['id']}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        # 4. Confirm deleted
        resp2 = await client.get("/api/history")
        records2 = resp2.json()
        assert not any(r["id"] == hist["id"] for r in records2)


@pytest.mark.asyncio
async def test_download_history_file_success_and_failures(tmp_path):
    await db.init_db()
    proj = await db.create_project(name="History DL Proj", video_path="dl_test.mp4")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Case 1: Non-existent history ID -> 404
        r_404 = await client.get("/api/history/non-existent-id/download")
        assert r_404.status_code == 404
        assert "Không tìm thấy bản ghi lịch sử" in r_404.json()["detail"]

        # Case 2: History with export_srt_path is None -> 404
        rec_none = await db.add_history_record(
            project_id=proj.id,
            mode="stt",
            export_srt_path=None,
        )
        r_none = await client.get(f"/api/history/{rec_none['id']}/download")
        assert r_none.status_code == 404
        assert "không có file phụ đề" in r_none.json()["detail"]

        # Case 3: History with deleted / non-existent file -> 404
        deleted_file = tmp_path / "deleted.srt"
        rec_deleted = await db.add_history_record(
            project_id=proj.id,
            mode="stt",
            export_srt_path=str(deleted_file),
        )
        r_del = await client.get(f"/api/history/{rec_deleted['id']}/download")
        assert r_del.status_code == 404
        assert "File phụ đề đã bị xóa hoặc không tồn tại" in r_del.json()["detail"]

        # Case 4: Valid subtitle file with Unicode Vietnamese filename -> 200 without 500 error
        unicode_file = tmp_path / "tiếng_việt_chính_xác.srt"
        unicode_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8")
        rec_unicode = await db.add_history_record(
            project_id=proj.id,
            mode="stt",
            export_srt_path=str(unicode_file),
        )
        r_valid = await client.get(f"/api/history/{rec_unicode['id']}/download")
        assert r_valid.status_code == 200
        assert "application/x-subrip" in r_valid.headers["content-type"]
        assert "attachment" in r_valid.headers["content-disposition"]
        assert "filename*=" in r_valid.headers["content-disposition"]
        assert "Xin chào" in r_valid.text
