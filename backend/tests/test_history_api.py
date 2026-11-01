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
