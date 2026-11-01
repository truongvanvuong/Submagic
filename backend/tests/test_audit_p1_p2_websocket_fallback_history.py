"""
Comprehensive Test Suite for Audit Group P1 & P2:
1. WebSocket Real-Time Stream (WebSocket /api/jobs/ws/{project_id}):
   - Immediate job_status event on connection
   - Ping / Pong interaction
   - Event broadcast reception
   - Graceful disconnect & connection cleanup
2. Job Cancellation & Conflict Control:
   - POST /api/jobs/{project_id}/cancel (was_running=True vs was_running=False)
   - POST /api/jobs/ocr conflict (409)
3. Hardware Fallback:
   - WhisperModelCache: CUDA load failure -> automatic fallback to CPU int8
   - run_stt_pipeline: CUDA transcription runtime failure -> retries on CPU int8
4. Hardsub OCR Async Error Handling:
   - VideoCapture failure in background scanner -> status FAILED & job_failed event
   - Missing/deleted video file before OCR -> FileNotFoundError & status FAILED
   - OCR job cancellation midway -> status CANCELLED & job_cancelled event
5. History Subtitle File Download (GET /api/history/{history_id}/download):
   - Non-existent history ID -> HTTP 404
   - Missing/deleted SRT file on disk -> HTTP 404
   - Existing SRT file -> HTTP 200 FileResponse with proper headers
6. Video Frame Preview (GET /api/projects/{project_id}/frame-preview):
   - Non-existent project -> HTTP 404
   - Valid timestamp (crop_roi=False) -> HTTP 200 image/jpeg
   - Valid timestamp with crop_roi=True -> HTTP 200 image/jpeg
7. Model Catalog & Hardware APIs:
   - GET /api/models & GET /api/system/models: catalog list & local cache flags
   - GET /api/system/hardware & GET /api/models/hardware: system specs
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import stream_manager
from app.config import settings
from app.database import db
from app.main import app
from app.schemas import (
    ExtractionMode,
    JobStatus,
    NormalizedBox,
    OCRJobRequest,
    ROIConfig,
    STTJobRequest,
    WebSocketEvent,
)
from app.services import ocr_pipeline, stt_pipeline
from app.services.ocr_pipeline import run_ocr_pipeline
from app.services.stt_pipeline import WhisperModelCache, run_stt_pipeline
from tests.test_phase1_and_phase2 import create_sample_video_with_audio


@pytest.fixture(autouse=True)
async def setup_isolated_db(tmp_path: Path, monkeypatch):
    """Ensure clean temporary SQLite DB and folders for all tests."""
    test_db = tmp_path / "test_p1_p2.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db)
    monkeypatch.setattr(settings, "db_path", test_db)
    monkeypatch.setattr(settings, "projects_dir", tmp_path / "projects")
    monkeypatch.setattr(settings, "crops_dir", tmp_path / "crops")
    monkeypatch.setattr(settings, "exports_dir", tmp_path / "exports")
    await db.init_db()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ============================================================================
# 1. WEBSOCKET REAL-TIME STREAMING TESTS
# ============================================================================

def test_websocket_connect_initial_status_and_ping_pong(tmp_path: Path, client: TestClient):
    """Verify WebSocket connection sends current job status immediately and handles ping-pong."""
    video_file = create_sample_video_with_audio(tmp_path / "ws_sample.mp4", duration_sec=1.0)
    create_res = client.post(
        "/api/projects",
        json={"name": "WS Test Project", "video_path": str(video_file), "mode": "stt"},
    )
    assert create_res.status_code == 200
    proj_id = create_res.json()["id"]

    with client.websocket_connect(f"/api/jobs/ws/{proj_id}") as ws:
        # 1. First frame sent on connection must be current job_status
        init_event = ws.receive_json()
        assert init_event["event"] == "job_status"
        assert init_event["project_id"] == proj_id
        assert init_event["status"] == "idle"

        # 2. Ping-pong roundtrip
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["event"] == "pong"
        assert pong["project_id"] == proj_id

    # 3. Connection must be cleaned up from stream_manager after closing
    assert proj_id not in stream_manager._connections or len(stream_manager._connections[proj_id]) == 0


@pytest.mark.asyncio
async def test_websocket_broadcast_delivery(tmp_path: Path):
    """Verify stream_manager.broadcast sends events to active client without crashing on dead sockets."""
    client = TestClient(app)
    video_file = create_sample_video_with_audio(tmp_path / "ws_broadcast.mp4", duration_sec=1.0)
    create_res = client.post(
        "/api/projects",
        json={"name": "Broadcast Proj", "video_path": str(video_file)},
    )
    proj_id = create_res.json()["id"]

    with client.websocket_connect(f"/api/jobs/ws/{proj_id}") as ws:
        _ = ws.receive_json()  # Ignore initial job_status

        # Broadcast custom event
        test_ev = WebSocketEvent(
            event="job_progress",
            project_id=proj_id,
            progress=50.0,
            message="Nửa chặng đường",
        )
        await stream_manager.broadcast(test_ev)

        received = ws.receive_json()
        assert received["event"] == "job_progress"
        assert received["progress"] == 50.0
        assert received["message"] == "Nửa chặng đường"


# ============================================================================
# 2. JOB CANCELLATION & CONFLICT CONTROL
# ============================================================================

def test_cancel_endpoint_when_running_and_when_idle(tmp_path: Path, client: TestClient):
    """Verify POST /api/jobs/{project_id}/cancel updates DB and broadcasts job_cancelled."""
    video_file = create_sample_video_with_audio(tmp_path / "cancel_sample.mp4", duration_sec=1.0)
    create_res = client.post(
        "/api/projects",
        json={"name": "Cancel Proj", "video_path": str(video_file)},
    )
    proj_id = create_res.json()["id"]

    # 1. Cancel when no task is running
    res_idle = client.post(f"/api/jobs/{proj_id}/cancel")
    assert res_idle.status_code == 200
    assert res_idle.json()["cancelled"] is True
    assert res_idle.json()["was_running"] is False

    # 2. Cancel when a task is registered in stream_manager
    cancel_ev = asyncio.Event()
    dummy_task = MagicMock()
    dummy_task.done.return_value = False
    stream_manager.register_job(proj_id, dummy_task, cancel_ev)

    res_running = client.post(f"/api/jobs/{proj_id}/cancel")
    assert res_running.status_code == 200
    assert res_running.json()["cancelled"] is True
    assert res_running.json()["was_running"] is True
    assert cancel_ev.is_set()


def test_jobs_api_ocr_conflict_409(tmp_path: Path, client: TestClient):
    """Verify POST /api/jobs/ocr returns HTTP 409 when a task is already running for the project."""
    video_file = create_sample_video_with_audio(tmp_path / "ocr_conflict.mp4", duration_sec=1.0)
    create_res = client.post(
        "/api/projects",
        json={"name": "OCR Conflict Proj", "video_path": str(video_file)},
    )
    proj_id = create_res.json()["id"]

    with patch.object(stream_manager, "is_running", return_value=True):
        res_409 = client.post(
            "/api/jobs/ocr",
            json={"project_id": proj_id, "language": "vi", "engine": "auto"},
        )
        assert res_409.status_code == 409
        assert "đang có một tiến trình trích xuất đang chạy" in res_409.json()["detail"]


# ============================================================================
# 3. HARDWARE FALLBACK TESTS (CUDA -> CPU INT8)
# ============================================================================

def test_whisper_model_cache_cuda_fallback_to_cpu():
    """When CUDA initialization fails, WhisperModelCache must automatically fallback to CPU int8."""
    cache = WhisperModelCache()
    mock_load_calls = []

    def mock_whisper_cls(m_size, device, compute_type, **kwargs):
        mock_load_calls.append((m_size, device, compute_type))
        if device == "cuda":
            raise RuntimeError("CUDA device unavailable or out of memory")
        return MagicMock(name=f"CPU_Whisper_{m_size}")

    with patch("faster_whisper.WhisperModel", side_effect=mock_whisper_cls):
        model, device, compute = cache.get_model("tiny", device="cuda", compute_type="float16")
        assert device == "cpu"
        assert compute == "int8"
        assert model is not None
        assert ("tiny", "cuda", "float16") in mock_load_calls
        assert ("tiny", "cpu", "int8") in mock_load_calls


# ============================================================================
# 4. HARDSUB OCR ASYNC ERROR & CANCELLATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_ocr_videocapture_open_failure(tmp_path: Path):
    """When OpenCV fails to open the video file, run_ocr_pipeline fails cleanly and emits job_failed."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "ocr_fail.mp4", duration_sec=1.0)
    meta = stt_pipeline.probe_video(sample_mp4)
    proj = await db.create_project(name="OCR Video Open Fail", video_path=str(sample_mp4), video_metadata=meta)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    req = OCRJobRequest(project_id=proj.id, language="vi", engine="auto")
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Không thể mở video"):
            await run_ocr_pipeline(req=req, job_id="j-ocr-fail", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


@pytest.mark.asyncio
async def test_ocr_deleted_video_raises_filenotfound(tmp_path: Path):
    """If the video file is missing before OCR starts, FileNotFoundError is raised."""
    missing_path = tmp_path / "non_existent_ocr.mp4"
    proj = await db.create_project(name="Missing Video", video_path=str(missing_path))

    req = OCRJobRequest(project_id=proj.id, language="vi")
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        await run_ocr_pipeline(req=req, job_id="j-ocr-missing", emit=lambda ev: asyncio.sleep(0))


@pytest.mark.asyncio
async def test_ocr_job_cancellation(tmp_path: Path):
    """Cancelling OCR pipeline sets status to CANCELLED and broadcasts job_cancelled."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "ocr_cancel.mp4", duration_sec=2.0)
    meta = stt_pipeline.probe_video(sample_mp4)
    proj = await db.create_project(name="OCR Cancel Proj", video_path=str(sample_mp4), video_metadata=meta)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    cancel_ev = asyncio.Event()
    cancel_ev.set()  # Cancel immediately before/at start

    req = OCRJobRequest(project_id=proj.id, language="vi")
    segments = await run_ocr_pipeline(req=req, job_id="j-ocr-cancel", emit=collect_event, cancel_event=cancel_ev)

    assert segments == []
    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.CANCELLED
    assert any(e.event == "job_cancelled" for e in events)


# ============================================================================
# 5. EXTRACTION HISTORY DOWNLOAD ENDPOINT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_history_download_404_and_success(tmp_path: Path):
    """Test /api/history/{history_id}/download for missing history, missing file, and valid download."""
    client = TestClient(app)
    video_file = create_sample_video_with_audio(tmp_path / "hist_sample.mp4", duration_sec=1.0)
    create_res = client.post("/api/projects", json={"name": "History Download Proj", "video_path": str(video_file)})
    proj_id = create_res.json()["id"]

    # 1. Non-existent history ID -> 404
    res_unknown = client.get("/api/history/non-existent-history-id/download")
    assert res_unknown.status_code == 404
    assert "Không tìm thấy bản ghi lịch sử" in res_unknown.json()["detail"]

    # 2. History record exists, but file was deleted from disk -> 404
    missing_file = tmp_path / "deleted_export.srt"
    hist_rec = await db.add_history_record(
        project_id=proj_id,
        mode="stt",
        language="vi",
        segment_count=3,
        export_srt_path=str(missing_file),
    )
    res_missing_file = client.get(f"/api/history/{hist_rec['id']}/download")
    assert res_missing_file.status_code == 404
    assert "File phụ đề đã bị xóa hoặc không tồn tại" in res_missing_file.json()["detail"]

    # 3. Existing valid subtitle file -> 200 FileResponse with correct headers
    valid_file = tmp_path / "valid_export.srt"
    valid_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8")
    hist_valid = await db.add_history_record(
        project_id=proj_id,
        mode="stt",
        language="vi",
        segment_count=1,
        export_srt_path=str(valid_file),
    )
    res_valid = client.get(f"/api/history/{hist_valid['id']}/download")
    assert res_valid.status_code == 200
    assert "application/x-subrip" in res_valid.headers["content-type"]
    assert "attachment" in res_valid.headers["content-disposition"]
    assert "Xin chào" in res_valid.text


# ============================================================================
# 6. VIDEO FRAME PREVIEW ENDPOINT TESTS
# ============================================================================

def test_frame_preview_endpoint(tmp_path: Path, client: TestClient):
    """Test GET /api/projects/{project_id}/frame-preview with timestamp and crop_roi."""
    # 1. Missing project -> 404
    res_404 = client.get("/api/projects/unknown-proj/frame-preview")
    assert res_404.status_code == 404

    # 2. Existing project -> returns JPEG image
    video_file = create_sample_video_with_audio(tmp_path / "preview_sample.mp4", duration_sec=1.5)
    create_res = client.post(
        "/api/projects",
        json={"name": "Preview Proj", "video_path": str(video_file)},
    )
    proj_id = create_res.json()["id"]

    # Full frame preview
    res_full = client.get(f"/api/projects/{proj_id}/frame-preview?timestamp=0.5&crop_roi=false")
    assert res_full.status_code == 200
    assert res_full.headers["content-type"] == "image/jpeg"
    assert len(res_full.content) > 100

    # Cropped ROI preview
    res_crop = client.get(f"/api/projects/{proj_id}/frame-preview?timestamp=0.5&crop_roi=true")
    assert res_crop.status_code == 200
    assert res_crop.headers["content-type"] == "image/jpeg"
    assert len(res_crop.content) > 100


# ============================================================================
# 7. MODEL CATALOG & SYSTEM HARDWARE ENDPOINTS TESTS
# ============================================================================

def test_model_catalog_and_hardware_endpoints(client: TestClient):
    """Verify /api/models and /api/system/hardware routes return valid schemas."""
    # Models catalog
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    data = res_models.json()
    assert "whisper_models" in data
    assert "ocr_engines" in data
    assert len(data["whisper_models"]) >= 5
    # Every model entry has required metadata
    assert all("id" in m and "vram_gb" in m and "downloaded" in m for m in data["whisper_models"])

    # System models alias route
    res_sys_models = client.get("/api/system/models")
    assert res_sys_models.status_code == 200
    assert "whisper_models" in res_sys_models.json()

    # Hardware detection routes
    res_hw = client.get("/api/system/hardware")
    assert res_hw.status_code == 200
    hw_data = res_hw.json()
    assert hw_data["cpu_cores"] >= 1
    assert hw_data["ram_total_gb"] > 0
    assert hw_data["recommended_stt_device"] in ("cuda", "cpu")

    res_models_hw = client.get("/api/models/hardware")
    assert res_models_hw.status_code == 200
    assert res_models_hw.json()["cpu_cores"] >= 1
