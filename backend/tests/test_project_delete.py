import asyncio
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.database import db
from app.main import app
from app.routers.jobs import stream_manager
from app.schemas import ExtractionMode, JobStatus, VideoMetadata


@pytest.fixture(autouse=True)
async def setup_test_db():
    await db.init_db()


@pytest.mark.asyncio
async def test_delete_nonexistent_project_returns_404():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete("/api/projects/nonexistent-id-999")
        assert res.status_code == 404
        assert "Không tìm thấy" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_delete_running_project_returns_409():
    project = await db.create_project(
        name="Running Project",
        video_path="sample_running.mp4",
        mode=ExtractionMode.STT,
    )
    # Set project status to running
    await db.update_project_status(project.id, JobStatus.RUNNING)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete(f"/api/projects/{project.id}")
        assert res.status_code == 409
        assert "đang trong quá trình xử lý" in res.json().get("detail", "")

    # Cleanup status so we can delete it
    await db.update_project_status(project.id, JobStatus.COMPLETED)
    await db.delete_project(project.id)


@pytest.mark.asyncio
async def test_delete_project_with_files_and_cascade_success(tmp_path):
    # Create fake files inside settings.data_dir
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    fake_video = settings.projects_dir / "test_delete_video.mp4"
    fake_video.write_bytes(b"dummy video data")

    project = await db.create_project(
        name="Project To Delete",
        video_path=str(fake_video),
        mode=ExtractionMode.STT,
    )

    # Create crops folder, audio file, export file
    crops_folder = settings.crops_dir / project.id
    crops_folder.mkdir(parents=True, exist_ok=True)
    (crops_folder / "frame_001.jpg").write_text("fake frame")

    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    fake_audio = settings.audio_dir / f"{project.id}.wav"
    fake_audio.write_bytes(b"fake audio")

    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    fake_export = settings.exports_dir / f"export_{project.id[:8]}.srt"
    fake_export.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

    # Add extraction history record
    await db.add_history_record(
        project_id=project.id,
        mode="stt",
        segment_count=1,
        export_srt_path=str(fake_export),
        duration_sec=10.0,
    )

    # Verify files and DB records exist before delete
    assert fake_video.exists()
    assert crops_folder.exists()
    assert fake_audio.exists()
    assert fake_export.exists()

    history_before = await db.list_history()
    assert any(h["project_id"] == project.id for h in history_before)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete(f"/api/projects/{project.id}")
        assert res.status_code == 200
        assert res.json() == {"deleted": True, "project_id": project.id}

    # Verify project deleted from DB
    deleted_proj = await db.get_project(project.id)
    assert deleted_proj is None

    # Verify CASCADE deleted history
    history_after = await db.list_history()
    assert not any(h["project_id"] == project.id for h in history_after)

    # Verify files deleted from disk
    assert not fake_video.exists()
    assert not crops_folder.exists()
    assert not fake_audio.exists()
    assert not fake_export.exists()


@pytest.mark.asyncio
async def test_delete_project_missing_files_succeeds():
    # Project with nonexistent video path
    project = await db.create_project(
        name="Missing Files Project",
        video_path=str(settings.projects_dir / "already_missing.mp4"),
        mode=ExtractionMode.STT,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete(f"/api/projects/{project.id}")
        assert res.status_code == 200
        assert res.json() == {"deleted": True, "project_id": project.id}

    deleted_proj = await db.get_project(project.id)
    assert deleted_proj is None


@pytest.mark.asyncio
async def test_bulk_delete_empty_list_returns_400():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/projects/bulk-delete", json={"project_ids": []})
        assert res.status_code == 400
        assert "không được để trống" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_bulk_delete_duplicate_ids_and_multiple_valid():
    p1 = await db.create_project(name="Dup 1", video_path="v1.mp4")
    p2 = await db.create_project(name="Dup 2", video_path="v2.mp4")
    p3 = await db.create_project(name="Dup 3", video_path="v3.mp4")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Pass duplicated IDs
        payload = {"project_ids": [p1.id, p1.id, p2.id, p1.id, p2.id, p3.id, p3.id]}
        res = await client.post("/api/projects/bulk-delete", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["deleted_count"] == 3
        assert set(data["deleted"]) == {p1.id, p2.id, p3.id}
        assert data["not_found_count"] == 0
        assert len(data["failed"]) == 0

    assert await db.get_project(p1.id) is None
    assert await db.get_project(p2.id) is None
    assert await db.get_project(p3.id) is None


@pytest.mark.asyncio
async def test_bulk_delete_projects_partial_and_running():
    p1 = await db.create_project(name="Bulk 1", video_path="v1.mp4")
    p2 = await db.create_project(name="Bulk 2", video_path="v2.mp4")
    p3 = await db.create_project(name="Bulk 3", video_path="v3.mp4")
    await db.update_project_status(p3.id, JobStatus.RUNNING)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"project_ids": [p1.id, p2.id, p3.id, "nonexistent-99"]}
        res = await client.post("/api/projects/bulk-delete", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["deleted_count"] == 2
        assert p1.id in data["deleted"]
        assert p2.id in data["deleted"]
        assert data["not_found_count"] == 1

        failed_map = {f["project_id"]: f["reason"] for f in data["failed"]}
        assert p3.id in failed_map
        assert "đang trong quá trình xử lý" in failed_map[p3.id]

    # Verify p1, p2 are deleted from DB, p3 remains
    assert await db.get_project(p1.id) is None
    assert await db.get_project(p2.id) is None
    assert await db.get_project(p3.id) is not None

    # Cleanup p3
    await db.update_project_status(p3.id, JobStatus.COMPLETED)
    await db.delete_project(p3.id)


@pytest.mark.asyncio
async def test_bulk_delete_nonexistent_ids():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/projects/bulk-delete",
            json={"project_ids": ["fake-id-1", "fake-id-2"]},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["deleted_count"] == 0
        assert data["not_found_count"] == 2
        assert len(data["failed"]) == 0
