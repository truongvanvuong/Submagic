"""Automated tests for Phase 1 (Backend Infrastructure, SQLite, Hardware, FFmpeg) and Phase 2 (STT Pipeline)."""
from __future__ import annotations

import asyncio
import math
import struct
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.database import DatabaseManager
from app.main import app
from app.schemas import (
    BilingualLayout,
    ExportFormat,
    ExtractionMode,
    NormalizedBox,
    ROIConfig,
    STTJobRequest,
    SubtitleSegment,
    SubtitleSegmentUpdate,
    WebSocketEvent,
    WordTimestamp,
)
from app.services.ffmpeg_service import (
    crop_and_mask_roi,
    extract_audio_for_stt,
    extract_frame_jpeg,
    probe_video,
)
from app.services.hardware import (
    detect_hardware,
    resolve_ffmpeg_executable,
    resolve_stt_device_and_compute,
)
from app.services.stt_pipeline import (
    SmartWordSegmenter,
    is_likely_hallucination,
    run_stt_pipeline,
    wrap_subtitle_lines,
)
from app.utils.subtitle_formatter import format_subtitles


def create_sample_video_with_audio(dest_path: Path, duration_sec: float = 2.0) -> Path:
    """Create a real MP4 video with a sine-wave audio track using FFmpeg or PyAV."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=640x360:rate=25",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={duration_sec}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(dest_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and dest_path.exists():
            return dest_path

    # Fallback using OpenCV if ffmpeg lavfi fails
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest_path), fourcc, 25.0, (640, 360))
    for i in range(int(duration_sec * 25)):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Sub Magic Frame {i}",
            (120, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
    writer.release()
    return dest_path


def test_hardware_detection():
    hw = detect_hardware()
    assert hw.cpu_cores >= 1
    assert hw.ram_total_gb > 0
    assert hw.recommended_stt_device in ("cuda", "cpu")
    device, compute_type = resolve_stt_device_and_compute("auto", "auto")
    assert device in ("cuda", "cpu")
    assert compute_type in ("float16", "int8")


@pytest.mark.asyncio
async def test_sqlite_database_lifecycle(tmp_path: Path):
    test_db_path = tmp_path / "test_sub_magic.sqlite3"
    manager = DatabaseManager(db_path=test_db_path)
    await manager.init_db()

    proj = await manager.create_project(
        name="Demo Video",
        video_path="C:/videos/demo.mp4",
        mode=ExtractionMode.STT,
        language="vi",
    )
    assert proj.id
    assert proj.name == "Demo Video"

    # Insert 2 segments (real-time auto-save simulation)
    seg1 = await manager.insert_segment(
        SubtitleSegment(
            project_id=proj.id,
            index=1,
            start_time=0.5,
            end_time=2.2,
            text="Xin chào các bạn,",
            confidence=0.95,
            source="stt",
        )
    )
    seg2 = await manager.insert_segment(
        SubtitleSegment(
            project_id=proj.id,
            index=2,
            start_time=2.3,
            end_time=4.5,
            text="chào mừng đến với Sub Magic.",
            confidence=0.92,
            source="stt",
        )
    )
    assert seg1.id is not None
    assert seg2.id is not None

    loaded = await manager.get_project(proj.id)
    assert loaded is not None
    assert len(loaded.segments) == 2

    # Update segment
    updated = await manager.update_segment(
        proj.id,
        seg1.id,
        SubtitleSegmentUpdate(
            text="Xin chào tất cả các bạn,",
            translated_text="Hello everyone,",
        ),
    )
    assert updated is not None
    assert updated.text == "Xin chào tất cả các bạn,"
    assert updated.translated_text == "Hello everyone,"

    # Merge two segments
    merged = await manager.merge_segments(proj.id, seg1.id, seg2.id)
    assert merged is not None
    assert "Xin chào tất cả các bạn, chào mừng đến với Sub Magic." in merged.text
    assert merged.start_time == 0.5
    assert merged.end_time == 4.5

    loaded_after = await manager.get_project(proj.id)
    assert loaded_after is not None
    assert len(loaded_after.segments) == 1


def test_ffmpeg_service_video_probe_and_audio_extraction(tmp_path: Path):
    video_file = create_sample_video_with_audio(tmp_path / "sample.mp4", duration_sec=2.0)
    meta = probe_video(video_file)
    assert meta.width == 640
    assert meta.height == 360
    assert meta.duration_sec >= 1.5

    if meta.has_audio:
        wav_path = extract_audio_for_stt(
            video_file,
            output_wav_path=tmp_path / "extracted_16k.wav",
            normalize_audio=True,
        )
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2

    roi_cfg = ROIConfig(
        main_roi=NormalizedBox(x=0.1, y=0.7, w=0.8, h=0.25),
        exclusion_masks=[NormalizedBox(x=0.75, y=0.72, w=0.12, h=0.10)],
    )
    jpg_bytes = extract_frame_jpeg(video_file, timestamp_sec=0.5, roi_config=roi_cfg)
    assert len(jpg_bytes) > 200


def test_smart_word_segmenter_and_line_wrapping():
    words = [
        WordTimestamp(word="Hôm", start=0.0, end=0.2, probability=0.99),
        WordTimestamp(word=" nay", start=0.2, end=0.4, probability=0.98),
        WordTimestamp(word=" chúng", start=0.4, end=0.6, probability=0.97),
        WordTimestamp(word=" ta", start=0.6, end=0.8, probability=0.99),
        WordTimestamp(word=" sẽ", start=0.8, end=1.0, probability=0.99),
        WordTimestamp(word=" xây", start=1.0, end=1.2, probability=0.99),
        WordTimestamp(word=" dựng", start=1.2, end=1.4, probability=0.99),
        WordTimestamp(word=" ứng", start=1.4, end=1.6, probability=0.99),
        WordTimestamp(word=" dụng", start=1.6, end=1.8, probability=0.99),
        WordTimestamp(word=" trích", start=1.8, end=2.0, probability=0.99),
        WordTimestamp(word=" xuất", start=2.0, end=2.2, probability=0.99),
        WordTimestamp(word=" phụ", start=2.2, end=2.4, probability=0.99),
        WordTimestamp(word=" đề.", start=2.4, end=2.7, probability=0.99),
        # Pause > 0.45s before next sentence
        WordTimestamp(word=" Chế", start=3.4, end=3.6, probability=0.95),
        WordTimestamp(word=" độ", start=3.6, end=3.8, probability=0.95),
        WordTimestamp(word=" hai", start=3.8, end=4.0, probability=0.95),
        WordTimestamp(word=" là", start=4.0, end=4.2, probability=0.95),
        WordTimestamp(word=" Hardsub", start=4.2, end=4.5, probability=0.95),
        WordTimestamp(word=" OCR!", start=4.5, end=4.9, probability=0.95),
    ]

    segmenter = SmartWordSegmenter(
        max_chars_per_line=28, max_lines_per_segment=2, max_pause_sec=0.45
    )
    segs = segmenter.segment_words(words, project_id="proj-1")
    assert len(segs) == 2
    assert segs[0].start_time == 0.0
    assert segs[0].end_time == 2.7
    assert "\n" in segs[0].text  # Wrapped across 2 balanced lines
    assert segs[1].start_time == 3.4
    assert segs[1].end_time == 4.9

    # Hallucination check
    assert is_likely_hallucination("la la la la la la la", 0.5) is True
    assert is_likely_hallucination("Hãy subscribe kênh để ủng hộ", 0.2) is True
    assert is_likely_hallucination("Câu thoại bình thường trong phim.", 0.9) is False


def test_subtitle_formatter_srt_vtt_txt_bilingual():
    segs = [
        SubtitleSegment(
            project_id="p1",
            index=1,
            start_time=1.25,
            end_time=3.5,
            text="Xin chào thế giới",
            translated_text="Hello world",
        )
    ]
    srt = format_subtitles(segs, ExportFormat.SRT, BilingualLayout.ORIGINAL_TOP)
    assert "00:00:01,250 --> 00:00:03,500" in srt
    assert "Xin chào thế giới\nHello world" in srt

    vtt = format_subtitles(segs, ExportFormat.VTT, BilingualLayout.TRANSLATED_TOP)
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.250 --> 00:00:03.500" in vtt
    assert "Hello world\nXin chào thế giới" in vtt

    txt = format_subtitles(
        segs, ExportFormat.TXT, BilingualLayout.NONE, include_timestamps_in_txt=True
    )
    assert "[00:00:01.250 -> 00:00:03.500] Xin chào thế giới" in txt


def test_fastapi_endpoints(tmp_path: Path):
    video_file = create_sample_video_with_audio(tmp_path / "api_test.mp4", duration_sec=1.5)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200

        hw = client.get("/api/system/hardware")
        assert hw.status_code == 200
        assert "recommended_stt_device" in hw.json()

        create_res = client.post(
            "/api/projects",
            json={
                "name": "API Test Video",
                "video_path": str(video_file),
                "mode": "stt",
                "language": "vi",
            },
        )
        assert create_res.status_code == 200
        proj_id = create_res.json()["id"]

        seg_res = client.post(
            f"/api/projects/{proj_id}/segments",
            json={"start_time": 0.0, "end_time": 1.2, "text": "Thử nghiệm xuất SRT"},
        )
        assert seg_res.status_code == 200

        exp_res = client.post(
            f"/api/projects/{proj_id}/export",
            json={"format": "srt", "bilingual_layout": "none"},
        )
        assert exp_res.status_code == 200
        assert "Thử nghiệm xuất SRT" in exp_res.text


@pytest.mark.asyncio
async def test_end_to_end_stt_pipeline_streaming_and_autosave(tmp_path: Path, monkeypatch):
    """Verify full run_stt_pipeline orchestration, audio extraction, smart segmentation, SQLite auto-save, and WebSocket events."""
    from app.database import db
    from app.services import stt_pipeline

    await db.init_db()
    video_file = create_sample_video_with_audio(tmp_path / "stt_e2e.mp4", duration_sec=3.0)
    meta = probe_video(video_file)
    project = await db.create_project(
        name="STT E2E Test",
        video_path=str(video_file),
        video_metadata=meta,
        mode=ExtractionMode.STT,
        language="vi",
    )

    class DummyWord:
        def __init__(self, word: str, start: float, end: float, probability: float = 0.96):
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class DummyRawSegment:
        def __init__(self):
            self.start = 0.1
            self.end = 2.8
            self.text = "Xin chào các bạn đến với Sub Magic."
            self.no_speech_prob = 0.02
            self.avg_logprob = -0.05
            self.words = [
                DummyWord("Xin", 0.1, 0.4),
                DummyWord(" chào", 0.4, 0.7),
                DummyWord(" các", 0.7, 1.0),
                DummyWord(" bạn.", 1.0, 1.3),
                DummyWord(" Chào", 1.9, 2.2),
                DummyWord(" mừng", 2.2, 2.5),
                DummyWord(" bạn!", 2.5, 2.8),
            ]

    class DummyWhisperModel:
        def transcribe(self, *args, **kwargs):
            return [DummyRawSegment()], {"language": "vi"}

    monkeypatch.setattr(
        stt_pipeline.model_cache,
        "get_model",
        lambda model_size, device, compute_type: (DummyWhisperModel(), "cpu", "int8"),
    )

    events: list[WebSocketEvent] = []

    async def collect_event(ev: WebSocketEvent) -> None:
        events.append(ev)

    req = STTJobRequest(
        project_id=project.id,
        model_size="small",
        language="vi",
        max_pause_sec=0.45,
    )
    segments = await run_stt_pipeline(req=req, job_id="job-test-1", emit=collect_event)
    assert len(segments) == 2
    assert segments[0].text == "Xin chào các bạn."
    assert segments[1].text == "Chào mừng bạn!"

    # Verify real-time events were emitted
    event_types = [e.event for e in events]
    assert "job_started" in event_types
    assert "segment_created" in event_types
    assert "job_completed" in event_types

    # Verify segments were auto-saved to SQLite
    saved_proj = await db.get_project(project.id)
    assert saved_proj is not None
    assert saved_proj.status.value == "completed"
    assert len(saved_proj.segments) == 2

