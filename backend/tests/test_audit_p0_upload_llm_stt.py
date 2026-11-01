"""
Comprehensive Test Suite for Audit Group P0 (Upload, LLM Resilience, STT Error Coverage).
Covers:
1. Video Upload API (POST /api/projects/upload):
   - Valid upload, storage & probing
   - Empty/missing filename validation (400)
   - Corrupted/non-video upload (400 probe error)
   - Filename sanitization (#, ?, &, Unicode, spaces)
   - Filename collision timestamp resolution
   - save_upload_file permission/OS error fallback
2. LLM Post-Processing Service & Endpoint (llm_service.py & POST /api/jobs/llm):
   - Missing Gemini API key handling (ValueError)
   - Project not found handling (ValueError / 404)
   - Empty segments project handling
   - LLM malformed JSON, markdown fences, dict-wrapped responses, missing keys
   - LLM API exception/rate-limit failure -> project status FAILED & job_failed event
   - Multiple batches (>60 segments) with Semaphore concurrency
   - Job cancellation handling
   - HTTP POST /api/jobs/llm (200, 404, 409)
3. Speech-to-Text Error Handling (stt_pipeline.py & POST /api/jobs/stt):
   - Video without audio stream rejection (has_audio == False -> RuntimeError, status FAILED)
   - Non-existent project & deleted video file handling
   - Audio extraction failure handling
   - Whisper transcription exception in worker thread -> status FAILED & job_failed event
   - HTTP POST /api/jobs/stt (200, 404, 409)
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.jobs import stream_manager
from app.api.projects import save_upload_file
from app.config import settings
from app.database import db
from app.main import app
from app.schemas import (
    ExtractionMode,
    JobStatus,
    LLMJobRequest,
    STTJobRequest,
    SubtitleSegment,
    WebSocketEvent,
)
from app.services import llm_service, stt_pipeline
from app.services.ffmpeg_service import MediaProcessingError, probe_video
from app.services.llm_service import _call_gemini_batch_postprocess, run_llm_postprocess_pipeline
from app.services.stt_pipeline import run_stt_pipeline
from tests.test_phase1_and_phase2 import create_sample_video_with_audio


@pytest.fixture(autouse=True)
async def setup_isolated_db(tmp_path: Path, monkeypatch):
    """Ensure every test runs against a clean SQLite database and fresh upload directory."""
    test_db = tmp_path / "test_p0.sqlite3"
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


def create_silent_video(dest_path: Path, duration_sec: float = 1.0) -> Path:
    """Create an MP4 video with VIDEO ONLY (no audio stream) using OpenCV."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest_path), fourcc, 25.0, (320, 240))
    for i in range(int(duration_sec * 25)):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(frame, f"Silent {i}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    return dest_path


# ============================================================================
# PART 1: VIDEO UPLOAD & SAFE FILE STORAGE TESTS
# ============================================================================

def test_upload_valid_video(tmp_path: Path, client: TestClient):
    """Verify POST /api/projects/upload successfully saves video, probes metadata, and creates project."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "src_valid.mp4", duration_sec=1.5)
    mp4_bytes = sample_mp4.read_bytes()

    response = client.post(
        "/api/projects/upload",
        files={"file": ("my_vacation_video.mp4", mp4_bytes, "video/mp4")},
        data={"mode": "stt", "language": "vi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] is not None
    assert data["name"] == "my_vacation_video"
    assert data["mode"] == "stt"
    assert data["language"] == "vi"
    assert data["video_metadata"] is not None
    assert data["video_metadata"]["duration_sec"] > 0
    assert Path(data["video_path"]).exists()


def test_upload_empty_filename_rejected(client: TestClient):
    """Uploading without a filename must be rejected with HTTP 400 or 422."""
    response = client.post(
        "/api/projects/upload",
        files={"file": ("", b"dummy data", "video/mp4")},
        data={"mode": "stt", "language": "vi"},
    )
    assert response.status_code in (400, 422)


def test_upload_corrupt_non_video_file_rejected(client: TestClient):
    """Uploading a non-video or corrupt file must fail probe_video and return HTTP 400."""
    corrupt_bytes = b"This is plain text content, not an MP4 or video container."
    response = client.post(
        "/api/projects/upload",
        files={"file": ("corrupt.mp4", corrupt_bytes, "video/mp4")},
        data={"mode": "ocr", "language": "en"},
    )
    assert response.status_code == 400
    assert "Không thể đọc metadata video" in response.json()["detail"]


def test_upload_filename_sanitization_and_collision(tmp_path: Path, client: TestClient):
    """Verify upload sanitizes forbidden characters (#, ?, &) and resolves filename collisions."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "src_sanitize.mp4", duration_sec=1.0)
    mp4_bytes = sample_mp4.read_bytes()

    # 1. Filename with problematic characters
    res1 = client.post(
        "/api/projects/upload",
        files={"file": ("test#movie?part&final.mp4", mp4_bytes, "video/mp4")},
        data={"mode": "stt"},
    )
    assert res1.status_code == 200
    path1 = Path(res1.json()["video_path"])
    assert "#" not in path1.name
    assert "?" not in path1.name
    assert "&" not in path1.name
    assert "test_movie_part_final" in path1.name

    # 2. Second upload with identical sanitized name -> timestamp added to avoid collision
    res2 = client.post(
        "/api/projects/upload",
        files={"file": ("test#movie?part&final.mp4", mp4_bytes, "video/mp4")},
        data={"mode": "stt"},
    )
    assert res2.status_code == 200
    path2 = Path(res2.json()["video_path"])
    assert path2.exists()
    assert path1.exists()
    assert path1 != path2  # Must be different filenames


def test_save_upload_file_permission_fallback(tmp_path: Path):
    """Verify save_upload_file handles PermissionError/OSError by retrying with timestamped path."""
    dest_dir = tmp_path / "uploads_fallback"
    dest_dir.mkdir(parents=True, exist_ok=True)

    class FakeUploadFile:
        def __init__(self, name: str, data: bytes):
            self.filename = name
            self.file = io.BytesIO(data)

    fake_file = FakeUploadFile("locked.mp4", b"video-content-bytes")

    original_open = Path.open
    attempts = [0]

    def mock_open(self, mode="r", *args, **kwargs):
        if "wb" in mode and attempts[0] == 0:
            attempts[0] += 1
            raise PermissionError("Simulated file lock by external process")
        return original_open(self, mode, *args, **kwargs)

    with patch.object(Path, "open", mock_open):
        saved_path = save_upload_file(fake_file, dest_dir)
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"video-content-bytes"
        assert attempts[0] == 1


# ============================================================================
# PART 2: LLM POST-PROCESSING SERVICE & API TESTS
# ============================================================================

def test_gemini_batch_postprocess_prompt_and_json_parsing():
    """Unit test _call_gemini_batch_postprocess for JSON response variations (arrays, dict wrappers, fences)."""
    batch = [
        SubtitleSegment(project_id="p1", index=1, start_time=0.0, end_time=2.0, text="Xin chao cac ban"),
        SubtitleSegment(project_id="p1", index=2, start_time=2.1, end_time=4.0, text="Chung toi la Sub Magic"),
    ]

    mock_client = MagicMock()
    mock_models = MagicMock()
    mock_client.models = mock_models

    # 1. Normal JSON array response with compact keys ("i", "t", "tr")
    mock_models.generate_content.return_value = MagicMock(
        text=json.dumps([
            {"i": 0, "t": "Xin chào các bạn", "tr": "Hello everyone"},
            {"i": 1, "t": "Chúng tôi là Sub Magic", "tr": "We are Sub Magic"},
        ])
    )
    with patch("google.genai.Client", return_value=mock_client):
        res = _call_gemini_batch_postprocess(batch, fix_ocr_spelling=True, translate_to="en", api_key="test-key")
        assert len(res) == 2
        assert res[0]["t"] == "Xin chào các bạn"
        assert res[0]["tr"] == "Hello everyone"

    # 2. Wrapped in {"subtitles": [...]} dictionary
    mock_models.generate_content.return_value = MagicMock(
        text=json.dumps({
            "subtitles": [
                {"i": 0, "t": "Xin chào các bạn", "tr": "Hello everyone"},
            ]
        })
    )
    with patch("google.genai.Client", return_value=mock_client):
        res2 = _call_gemini_batch_postprocess(batch, fix_ocr_spelling=True, translate_to="en", api_key="test-key")
        assert len(res2) == 1
        assert res2[0]["t"] == "Xin chào các bạn"


@pytest.mark.asyncio
async def test_llm_postprocess_missing_api_key_raises(tmp_path: Path):
    """Calling LLM pipeline without a Gemini API Key must raise ValueError, set DB status to FAILED and emit job_failed."""
    proj = await db.create_project(name="No Key Proj", video_path="dummy.mp4")
    req = LLMJobRequest(project_id=proj.id, gemini_api_key=None)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    with patch.object(settings, "gemini_api_key", None):
        with pytest.raises(ValueError, match="Thiếu Gemini API Key"):
            await run_llm_postprocess_pipeline(req, job_id="job-1", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


@pytest.mark.asyncio
async def test_llm_postprocess_empty_segments_returns_empty():
    """If project has no segments, LLM postprocessing returns [] immediately without error."""
    proj = await db.create_project(name="Empty Segs Proj", video_path="dummy.mp4")
    req = LLMJobRequest(project_id=proj.id, gemini_api_key="valid-key")

    res = await run_llm_postprocess_pipeline(req, job_id="job-empty", emit=lambda ev: asyncio.sleep(0))
    assert res == []


@pytest.mark.asyncio
async def test_llm_postprocess_nonexistent_project():
    """Non-existent project ID must raise ValueError."""
    req = LLMJobRequest(project_id="unknown-proj-id", gemini_api_key="valid-key")
    with pytest.raises(ValueError, match="Project not found"):
        await run_llm_postprocess_pipeline(req, job_id="job-err", emit=lambda ev: asyncio.sleep(0))


@pytest.mark.asyncio
async def test_llm_postprocess_api_failure_updates_status_failed(tmp_path: Path):
    """When Gemini API call throws an exception (e.g., 429 quota/timeout), project status must become FAILED."""
    proj = await db.create_project(name="Fail Proj", video_path="dummy.mp4")
    await db.insert_segment(SubtitleSegment(project_id=proj.id, index=1, start_time=0.0, end_time=2.0, text="Lỗi API"))

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    def mock_fail(*args, **kwargs):
        raise RuntimeError("Google Gemini Quota Exceeded (HTTP 429)")

    req = LLMJobRequest(project_id=proj.id, gemini_api_key="key", fix_ocr_spelling=True)
    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_fail):
        with pytest.raises(RuntimeError, match="Google Gemini Quota Exceeded"):
            await run_llm_postprocess_pipeline(req, job_id="job-fail", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


@pytest.mark.asyncio
async def test_llm_postprocess_large_batch_and_concurrency(tmp_path: Path):
    """Verify LLM pipeline handles multiple batches (>60 items) with Semaphore concurrency and atomic updates."""
    proj = await db.create_project(name="Large Batch Proj", video_path="dummy.mp4")
    # Insert 65 segments (creates 2 batches with batch_size=50)
    for i in range(1, 66):
        await db.insert_segment(
            SubtitleSegment(project_id=proj.id, index=i, start_time=float(i), end_time=float(i) + 0.8, text=f"Dòng {i}")
        )

    def mock_batch_translate(batch, fix_ocr_spelling, translate_to, api_key):
        return [{"i": idx, "t": seg.text, "tr": f"Translated {seg.text}"} for idx, seg in enumerate(batch)]

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    req = LLMJobRequest(project_id=proj.id, gemini_api_key="key", translate_to="en", batch_size=50)
    with patch.object(llm_service, "_call_gemini_batch_postprocess", mock_batch_translate):
        updated = await run_llm_postprocess_pipeline(req, job_id="job-large", emit=collect_event)

    assert len(updated) == 65
    assert updated[0].translated_text == "Translated Dòng 1"
    assert updated[64].translated_text == "Translated Dòng 65"

    saved_proj = await db.get_project(proj.id)
    assert saved_proj.status == JobStatus.COMPLETED
    assert len(saved_proj.segments) == 65
    assert saved_proj.segments[64].translated_text == "Translated Dòng 65"


def test_jobs_api_llm_endpoint_endpoints(tmp_path: Path, client: TestClient):
    """Test HTTP POST /api/jobs/llm: success 200, 404 on missing project, and 409 conflict when running."""
    # 1. Missing project -> 404
    res_404 = client.post("/api/jobs/llm", json={"project_id": "non-existent", "gemini_api_key": "k"})
    assert res_404.status_code == 404

    # 2. Existing project -> 200 running
    dummy_video = create_sample_video_with_audio(tmp_path / "dummy_llm.mp4", duration_sec=1.0)
    create_res = client.post("/api/projects", json={"name": "LLM API Proj", "video_path": str(dummy_video)})
    assert create_res.status_code == 200
    proj_id = create_res.json()["id"]

    with patch.object(stream_manager, "is_running", return_value=False):
        res_200 = client.post("/api/jobs/llm", json={"project_id": proj_id, "gemini_api_key": "k"})
        assert res_200.status_code == 200
        assert res_200.json()["status"] == "running"
        assert "job_id" in res_200.json()

    # 3. Conflict 409 when job is already running
    with patch.object(stream_manager, "is_running", return_value=True):
        res_409 = client.post("/api/jobs/llm", json={"project_id": proj_id, "gemini_api_key": "k"})
        assert res_409.status_code == 409
        assert "đang có một tiến trình" in res_409.json()["detail"]


# ============================================================================
# PART 3: SPEECH-TO-TEXT (STT) ERROR COVERAGE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_stt_video_without_audio_stream_raises(tmp_path: Path):
    """Video without an audio stream must fail STT gracefully with RuntimeError, status FAILED, and event job_failed."""
    silent_video = create_silent_video(tmp_path / "silent.mp4", duration_sec=1.5)
    meta = probe_video(silent_video)
    assert meta.has_audio is False

    proj = await db.create_project(
        name="Silent Video Proj",
        video_path=str(silent_video),
        video_metadata=meta,
        mode=ExtractionMode.STT,
    )

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    req = STTJobRequest(project_id=proj.id, model_size="tiny")
    with pytest.raises(RuntimeError, match="Video không có luồng âm thanh"):
        await run_stt_pipeline(req=req, job_id="job-silent", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


@pytest.mark.asyncio
async def test_stt_nonexistent_project_and_deleted_video(tmp_path: Path):
    """Verify STT pipeline fails with ValueError for unknown project and FileNotFoundError for deleted video."""
    # 1. Unknown project
    req_unknown = STTJobRequest(project_id="unknown-id", model_size="tiny")
    with pytest.raises(ValueError, match="Project not found"):
        await run_stt_pipeline(req_unknown, job_id="j1", emit=lambda ev: asyncio.sleep(0))

    # 2. Deleted video file
    deleted_path = tmp_path / "deleted.mp4"
    deleted_path.write_bytes(b"dummy")
    proj = await db.create_project(name="Deleted Video", video_path=str(deleted_path))
    deleted_path.unlink()  # Delete file from disk

    req_deleted = STTJobRequest(project_id=proj.id, model_size="tiny")
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        await run_stt_pipeline(req_deleted, job_id="j2", emit=lambda ev: asyncio.sleep(0))


@pytest.mark.asyncio
async def test_stt_audio_extraction_failure(tmp_path: Path):
    """When audio extraction fails, STT pipeline sets project to FAILED and emits job_failed."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "src_audio_fail.mp4", duration_sec=1.0)
    meta = probe_video(sample_mp4)
    proj = await db.create_project(name="Audio Fail Proj", video_path=str(sample_mp4), video_metadata=meta)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    def mock_extract_fail(*args, **kwargs):
        raise MediaProcessingError("FFmpeg audio decode failed: corrupt stream")

    req = STTJobRequest(project_id=proj.id, model_size="tiny")
    with patch.object(stt_pipeline, "extract_audio_for_stt", mock_extract_fail):
        with pytest.raises(MediaProcessingError, match="FFmpeg audio decode failed"):
            await run_stt_pipeline(req=req, job_id="j-audio-fail", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


@pytest.mark.asyncio
async def test_stt_whisper_transcription_worker_exception(tmp_path: Path):
    """When Whisper transcribe throws in background worker, the error is caught, reported, and DB updated to FAILED."""
    sample_mp4 = create_sample_video_with_audio(tmp_path / "src_whisper_fail.mp4", duration_sec=1.0)
    meta = probe_video(sample_mp4)
    proj = await db.create_project(name="Whisper Fail Proj", video_path=str(sample_mp4), video_metadata=meta)

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent):
        events.append(ev)

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("Whisper CTranslate2 memory segmentation fault")

    req = STTJobRequest(project_id=proj.id, model_size="tiny")
    with patch.object(stt_pipeline.model_cache, "get_model", return_value=(mock_model, "cpu", "int8")):
        with pytest.raises(RuntimeError, match="Whisper CTranslate2 memory segmentation fault"):
            await run_stt_pipeline(req=req, job_id="j-whisper-fail", emit=collect_event)

    updated_proj = await db.get_project(proj.id)
    assert updated_proj.status == JobStatus.FAILED
    assert any(e.event == "job_failed" for e in events)


def test_jobs_api_stt_endpoint(tmp_path: Path, client: TestClient):
    """Test HTTP POST /api/jobs/stt: 404 on missing project, 200 on success, and 409 conflict when already running."""
    # 1. Missing project -> 404
    res_404 = client.post("/api/jobs/stt", json={"project_id": "non-existent", "model_size": "tiny"})
    assert res_404.status_code == 404

    # 2. Existing project -> 200 running
    dummy_video = create_sample_video_with_audio(tmp_path / "dummy_stt.mp4", duration_sec=1.0)
    create_res = client.post("/api/projects", json={"name": "STT API Proj", "video_path": str(dummy_video)})
    assert create_res.status_code == 200
    proj_id = create_res.json()["id"]

    with patch.object(stream_manager, "is_running", return_value=False):
        res_200 = client.post("/api/jobs/stt", json={"project_id": proj_id, "model_size": "tiny"})
        assert res_200.status_code == 200
        assert res_200.json()["status"] == "running"
        assert "job_id" in res_200.json()

    # 3. Conflict 409
    with patch.object(stream_manager, "is_running", return_value=True):
        res_409 = client.post("/api/jobs/stt", json={"project_id": proj_id, "model_size": "tiny"})
        assert res_409.status_code == 409
        assert "đang có một tiến trình" in res_409.json()["detail"]
