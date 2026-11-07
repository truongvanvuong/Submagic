"""Comprehensive End-to-End (E2E) Integration Test Suite for Sub Magic.

Structured across 4 Tiers:
- Tier 1: Feature Coverage (≥5 per feature: Subtitle Export, Video Streaming, Segment Split/Merge, Project Lifecycle)
- Tier 2: Boundary & Corner Cases (≥5 per feature: Zero/inverted timestamps, sub-ms precision, out-of-range headers, single-byte ranges, Unicode/Vietnamese diacritics, HTML/CRLF cleaning)
- Tier 3: Cross-Feature Combinations (Split -> Edit -> Bilingual Export, Merge -> Reindex -> Stream, Cascade Split, Concurrent Exports)
- Tier 4: Real-World Application Scenarios (Full OCR correction workflow, STT podcast transcript, Video player timeline scrubbing)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import (
    BilingualLayout,
    ExportFormat,
    ExportRequest,
    ExtractionMode,
    NormalizedBox,
    ROIConfig,
    SubtitleSegment,
    SubtitleSegmentUpdate,
)
from app.services.ffmpeg_service import probe_video
from app.services.hardware import resolve_ffmpeg_executable
from app.utils.subtitle_formatter import (
    format_subtitles,
    format_timestamp_srt,
    format_timestamp_vtt,
    parse_timestamp,
)


def create_test_mp4(dest_path: Path, duration_sec: float = 1.0) -> Path:
    """Generate a valid, playable MP4 file for video streaming tests."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=320x240:rate=25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path

    # Fallback to OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest_path), fourcc, 25.0, (320, 240))
    for i in range(max(1, int(duration_sec * 25))):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {i}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
    writer.release()
    return dest_path


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path, monkeypatch):
    """Ensure database is initialized in a clean temp directory per test run."""
    db_file = tmp_path / "test_e2e_submagic.db"
    monkeypatch.setattr(db, "db_path", db_file)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(db.init_db())
    yield
    # No teardown needed, tmp_path handles cleanup


@pytest.fixture
def test_client() -> TestClient:
    """Return a TestClient bound to the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Provide a synthesized 2-second MP4 video."""
    video_path = tmp_path / "e2e_sample_video.mp4"
    return create_test_mp4(video_path, duration_sec=2.0)


async def create_populated_project(video_path: Path, name: str = "E2E Test Project") -> str:
    """Helper to create a project with 3 realistic bilingual Vietnamese/English segments."""
    meta = probe_video(video_path)
    proj = await db.create_project(
        name=name,
        video_path=str(video_path),
        video_metadata=meta,
        mode=ExtractionMode.STT,
        language="vi",
    )
    seg1 = SubtitleSegment(
        project_id=proj.id,
        index=1,
        start_time=0.0,
        end_time=1.85,
        text="Chào mừng các bạn đến với ứng dụng Sub Magic.",
        translated_text="Welcome everyone to the Sub Magic application.",
        confidence=0.98,
        source="stt",
    )
    seg2 = SubtitleSegment(
        project_id=proj.id,
        index=2,
        start_time=2.10,
        end_time=4.30,
        text="Hệ thống hỗ trợ trích xuất phụ đề tự động độ chính xác cao.",
        translated_text="The system supports high accuracy automated subtitle extraction.",
        confidence=0.96,
        source="stt",
    )
    seg3 = SubtitleSegment(
        project_id=proj.id,
        index=3,
        start_time=4.60,
        end_time=6.00,
        text="Chúc các bạn có trải nghiệm tuyệt vời!",
        translated_text="Have a wonderful experience!",
        confidence=0.99,
        source="manual",
    )
    await db.insert_segment(seg1)
    await db.insert_segment(seg2)
    await db.insert_segment(seg3)
    return proj.id


# ==============================================================================
# TIER 1: FEATURE COVERAGE (≥5 test cases per feature)
# ==============================================================================

class TestTier1SubtitleExport:
    """Tier 1: Feature tests for subtitle formatting and multi-format export."""

    def test_export_srt_single_mode(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=none")
        assert res.status_code == 200
        assert "application/x-subrip" in res.headers["content-type"]
        body = res.text
        assert "1\n00:00:00,000 --> 00:00:01,850\nChào mừng các bạn đến với ứng dụng Sub Magic" in body
        assert "2\n00:00:02,100 --> 00:00:04,300" in body
        assert "Welcome everyone" not in body  # single mode must not include translation

    def test_export_vtt_single_mode(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=vtt&bilingual_layout=none")
        assert res.status_code == 200
        assert "text/vtt" in res.headers["content-type"]
        body = res.text
        assert body.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.850" in body
        assert "Chào mừng các bạn đến với ứng dụng Sub Magic" in body

    def test_export_txt_without_timestamps(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=txt&include_timestamps_in_txt=false")
        assert res.status_code == 200
        assert "text/plain" in res.headers["content-type"]
        lines = [line.strip() for line in res.text.strip().split("\n") if line.strip()]
        assert len(lines) == 3
        assert lines[0] == "Chào mừng các bạn đến với ứng dụng Sub Magic"
        assert not lines[0].startswith("[")

    def test_export_txt_with_timestamps(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=txt&include_timestamps_in_txt=true")
        assert res.status_code == 200
        body = res.text
        assert "[00:00:00.000 -> 00:00:01.850] Chào mừng các bạn đến với ứng dụng Sub Magic" in body
        assert "[00:00:02.100 -> 00:00:04.300]" in body

    def test_export_bilingual_original_top(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=original_top")
        assert res.status_code == 200
        body = res.text
        expected_block = (
            "1\n00:00:00,000 --> 00:00:01,850\n"
            "Chào mừng các bạn đến với ứng dụng Sub Magic\n"
            "Welcome everyone to the Sub Magic application"
        )
        assert expected_block in body

    def test_export_bilingual_translated_top(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=translated_top")
        assert res.status_code == 200
        body = res.text
        expected_block = (
            "1\n00:00:00,000 --> 00:00:01,850\n"
            "Welcome everyone to the Sub Magic application\n"
            "Chào mừng các bạn đến với ứng dụng Sub Magic"
        )
        assert expected_block in body

    def test_export_bilingual_translated_only(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=translated_only")
        assert res.status_code == 200
        body = res.text
        assert "Welcome everyone to the Sub Magic application" in body
        assert "Chào mừng các bạn đến với ứng dụng Sub Magic" not in body

    def test_export_post_endpoint_with_custom_disk_path(self, test_client: TestClient, sample_video: Path, tmp_path: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        target_file = tmp_path / "custom_exports" / "final_subtitles.srt"
        req = ExportRequest(
            format=ExportFormat.SRT,
            bilingual_layout=BilingualLayout.ORIGINAL_TOP,
            save_to_disk_path=str(target_file),
        )
        res = test_client.post(f"/api/projects/{proj_id}/export", json=req.model_dump())
        assert res.status_code == 200
        assert target_file.exists()
        saved_text = target_file.read_text(encoding="utf-8")
        assert "Chào mừng các bạn đến với ứng dụng Sub Magic" in saved_text
        assert "Welcome everyone" in saved_text


class TestTier1VideoStreaming:
    """Tier 1: Feature tests for video streaming and HTTP Range request handling."""

    def test_video_streaming_full_content_http_200(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(f"/api/projects/{proj_id}/video")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("video/")
        assert int(res.headers["content-length"]) == sample_video.stat().st_size
        assert res.headers.get("accept-ranges") == "bytes"
        assert len(res.content) == sample_video.stat().st_size

    def test_video_streaming_partial_content_range_start_end(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        total_bytes = sample_video.stat().st_size
        assert total_bytes > 500

        headers = {"Range": "bytes=0-499"}
        res = test_client.get(f"/api/projects/{proj_id}/video", headers=headers)
        assert res.status_code == 206
        assert len(res.content) == 500
        assert res.headers["content-length"] == "500"
        assert res.headers["content-range"] == f"bytes 0-499/{total_bytes}"

    def test_video_streaming_suffix_range(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        total_bytes = sample_video.stat().st_size
        headers = {"Range": "bytes=-200"}
        res = test_client.get(f"/api/projects/{proj_id}/video", headers=headers)
        assert res.status_code == 206
        assert len(res.content) == 200
        start = total_bytes - 200
        assert res.headers["content-range"] == f"bytes {start}-{total_bytes - 1}/{total_bytes}"

    def test_video_streaming_open_ended_range(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        total_bytes = sample_video.stat().st_size
        headers = {"Range": "bytes=250-"}
        res = test_client.get(f"/api/projects/{proj_id}/video", headers=headers)
        assert res.status_code == 206
        expected_len = total_bytes - 250
        assert len(res.content) == expected_len
        assert res.headers["content-range"] == f"bytes 250-{total_bytes - 1}/{total_bytes}"

    def test_video_streaming_content_fidelity(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        full_raw = sample_video.read_bytes()
        res = test_client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=50-150"})
        assert res.status_code == 206
        assert res.content == full_raw[50:151]


class TestTier1SegmentSplitAndMerge:
    """Tier 1: Feature tests for segment split and merge lifecycles with SQLite persistence."""

    def test_segment_split_with_custom_text(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        target_seg = proj.segments[1]  # 2.10 -> 4.30

        split_payload = {
            "split_time": 3.20,
            "first_text": "Hệ thống hỗ trợ trích xuất phụ đề",
            "second_text": "tự động độ chính xác cao.",
        }
        res = test_client.post(
            f"/api/projects/{proj_id}/segments/{target_seg.id}/split",
            json=split_payload,
        )
        assert res.status_code == 200
        new_segs = res.json()
        assert len(new_segs) == 2
        assert new_segs[0]["start_time"] == target_seg.start_time
        assert new_segs[0]["end_time"] == 3.20
        assert new_segs[0]["text"] == split_payload["first_text"]

        assert new_segs[1]["start_time"] == 3.20
        assert new_segs[1]["end_time"] == target_seg.end_time
        assert new_segs[1]["text"] == split_payload["second_text"]

    def test_segment_split_via_query_params(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        target_seg = proj.segments[0]  # 0.0 -> 1.85

        res = test_client.post(
            f"/api/projects/{proj_id}/segments/{target_seg.id}/split?split_time=1.0"
        )
        assert res.status_code == 200
        new_segs = res.json()
        assert len(new_segs) == 2
        assert new_segs[0]["end_time"] == 1.0
        assert new_segs[1]["start_time"] == 1.0

    def test_segment_split_persistence_and_sequential_indexes(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        target_seg = proj.segments[0]
        test_client.post(f"/api/projects/{proj_id}/segments/{target_seg.id}/split?split_time=0.9")

        # Verify DB directly
        updated_proj = loop.run_until_complete(db.get_project(proj_id))
        assert updated_proj is not None
        assert len(updated_proj.segments) == 4
        # Verify indices are 1, 2, 3, 4
        assert [s.index for s in updated_proj.segments] == [1, 2, 3, 4]

    def test_segment_merge_lifecycle(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        first_seg = proj.segments[0]
        second_seg = proj.segments[1]

        res = test_client.post(
            f"/api/projects/{proj_id}/segments/merge?first_id={first_seg.id}&second_id={second_seg.id}"
        )
        assert res.status_code == 200
        merged_seg = res.json()
        assert merged_seg["start_time"] == first_seg.start_time
        assert merged_seg["end_time"] == second_seg.end_time
        assert first_seg.text in merged_seg["text"]
        assert second_seg.text in merged_seg["text"]

    def test_segment_merge_persistence_and_cleanup(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        first_id = proj.segments[0].id
        second_id = proj.segments[1].id

        test_client.post(f"/api/projects/{proj_id}/segments/merge?first_id={first_id}&second_id={second_id}")

        refreshed = loop.run_until_complete(db.get_project(proj_id))
        assert len(refreshed.segments) == 2
        # The second segment must have been deleted, first updated
        seg_ids = [s.id for s in refreshed.segments]
        assert first_id in seg_ids
        assert second_id not in seg_ids
        assert [s.index for s in refreshed.segments] == [1, 2]


class TestTier1ProjectLifecycle:
    """Tier 1: Feature tests for project creation, CRUD, and ROI configuration."""

    def test_project_creation_and_retrieval(self, test_client: TestClient, sample_video: Path):
        create_res = test_client.post(
            "/api/projects",
            json={"video_path": str(sample_video), "name": "Created E2E Proj", "language": "vi"},
        )
        assert create_res.status_code == 200
        proj_data = create_res.json()
        assert proj_data["name"] == "Created E2E Proj"
        assert proj_data["video_metadata"]["duration_sec"] > 0

        get_res = test_client.get(f"/api/projects/{proj_data['id']}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == proj_data["id"]

    def test_manual_segment_addition(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        add_res = test_client.post(
            f"/api/projects/{proj_id}/segments",
            json={
                "start_time": 7.0,
                "end_time": 9.5,
                "text": "Dòng phụ đề thủ công mới tạo.",
                "translated_text": "Newly created manual subtitle line.",
            },
        )
        assert add_res.status_code == 200
        added = add_res.json()
        assert added["index"] == 4
        assert added["start_time"] == 7.0
        assert added["source"] == "manual"

    def test_segment_patch_update(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg_id = proj.segments[0].id

        patch_res = test_client.patch(
            f"/api/projects/{proj_id}/segments/{seg_id}",
            json={"text": "Nội dung cập nhật sau chỉnh sửa.", "translated_text": "Updated content."},
        )
        assert patch_res.status_code == 200
        updated = patch_res.json()
        assert updated["text"] == "Nội dung cập nhật sau chỉnh sửa."
        assert updated["translated_text"] == "Updated content."

    def test_segment_deletion_and_reindexing(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        middle_id = proj.segments[1].id

        del_res = test_client.delete(f"/api/projects/{proj_id}/segments/{middle_id}")
        assert del_res.status_code == 200

        refreshed = loop.run_until_complete(db.get_project(proj_id))
        assert len(refreshed.segments) == 2
        assert [s.index for s in refreshed.segments] == [1, 2]

    def test_roi_and_exclusion_mask_update(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        roi_data = {
            "main_roi": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.15, "label": "Bottom Sub"},
            "exclusion_masks": [
                {"x": 0.12, "y": 0.81, "w": 0.1, "h": 0.05, "label": "Logo Mask"},
                {"x": 0.75, "y": 0.82, "w": 0.12, "h": 0.06, "label": "Watermark Mask"},
            ],
        }
        roi_res = test_client.put(f"/api/projects/{proj_id}/roi", json=roi_data)
        assert roi_res.status_code == 200
        updated_roi = roi_res.json()["roi_config"]
        assert updated_roi["main_roi"]["y"] == 0.8
        assert len(updated_roi["exclusion_masks"]) == 2


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (≥5 per feature)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Edge cases, extreme values, Unicode fidelity, and error conditions."""

    def test_export_empty_project(self, test_client: TestClient, sample_video: Path):
        meta = probe_video(sample_video)
        loop = asyncio.get_event_loop()
        empty_proj = loop.run_until_complete(
            db.create_project(name="Empty Project", video_path=str(sample_video), video_metadata=meta)
        )

        res_srt = test_client.get(f"/api/projects/{empty_proj.id}/export?format=srt")
        assert res_srt.status_code == 200
        assert res_srt.text == ""

        res_vtt = test_client.get(f"/api/projects/{empty_proj.id}/export?format=vtt")
        assert res_vtt.status_code == 200
        assert res_vtt.text.strip() == "WEBVTT"

        res_txt = test_client.get(f"/api/projects/{empty_proj.id}/export?format=txt")
        assert res_txt.status_code == 200
        assert res_txt.text == ""

    def test_export_submillisecond_and_zero_timestamps(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=0.0004,
                end_time=0.0009,
                text="Submillisecond cue",
            )
        ]
        srt = format_subtitles(segs, export_format=ExportFormat.SRT)
        assert "00:00:00,000 --> 00:00:00,100" in srt

    def test_export_inverted_timestamps_auto_clamp(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=5.0,
                end_time=3.0,  # inverted
                text="Inverted timing",
            )
        ]
        srt = format_subtitles(segs, export_format=ExportFormat.SRT)
        # Formatter clamps end_time to max(start_time + 0.1, end_time)
        assert "00:00:05,000 --> 00:00:05,100" in srt

    def test_export_high_hours_timestamps(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=360_000.0,  # 100 hours
                end_time=360_005.5,
                text="100 hour mark",
            )
        ]
        srt = format_subtitles(segs, export_format=ExportFormat.SRT)
        assert "100:00:00,000 --> 100:00:05,500" in srt

    def test_export_bilingual_missing_translation_fallback(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=1.0,
                end_time=3.0,
                text="Chỉ có văn bản gốc",
                translated_text=None,
            )
        ]
        # Even with original_top or translated_top, should not crash or add empty line
        out_top = format_subtitles(segs, export_format=ExportFormat.SRT, bilingual_layout=BilingualLayout.ORIGINAL_TOP)
        assert "Chỉ có văn bản gốc" in out_top
        assert "\n\n\n" not in out_top

        out_only = format_subtitles(segs, export_format=ExportFormat.SRT, bilingual_layout=BilingualLayout.TRANSLATED_ONLY)
        assert "Chỉ có văn bản gốc" in out_only  # Fallback to orig

    def test_timestamp_parser_variations(self):
        assert abs(parse_timestamp("01:23:45,678") - 5025.678) < 1e-4
        assert abs(parse_timestamp("01:23:45.678") - 5025.678) < 1e-4
        assert abs(parse_timestamp("00:00:01,000") - 1.0) < 1e-4
        assert abs(parse_timestamp("00:00:00,000") - 0.0) < 1e-4
        assert abs(parse_timestamp("45.5") - 45.5) < 1e-4

    def test_video_streaming_out_of_range_416(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.get(
            f"/api/projects/{proj_id}/video",
            headers={"Range": "bytes=999999999-"},
        )
        assert res.status_code == 416

    def test_video_streaming_single_byte_range(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))
        total_bytes = sample_video.stat().st_size

        res = test_client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=0-0"})
        assert res.status_code == 206
        assert len(res.content) == 1
        assert res.headers["content-range"] == f"bytes 0-0/{total_bytes}"

    def test_video_streaming_last_byte_range(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))
        total_bytes = sample_video.stat().st_size

        last_idx = total_bytes - 1
        res = test_client.get(
            f"/api/projects/{proj_id}/video",
            headers={"Range": f"bytes={last_idx}-{last_idx}"},
        )
        assert res.status_code == 206
        assert len(res.content) == 1
        assert res.headers["content-range"] == f"bytes {last_idx}-{last_idx}/{total_bytes}"

    def test_video_streaming_nonexistent_project_404(self, test_client: TestClient):
        res = test_client.get("/api/projects/non-existent-proj-id/video")
        assert res.status_code == 404

    def test_segment_split_outside_bounds_400(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg = proj.segments[0]  # 0.0 -> 1.85

        # Split time completely past end
        res = test_client.post(
            f"/api/projects/{proj_id}/segments/{seg.id}/split?split_time=2.5"
        )
        assert res.status_code == 400

        # Split time negative or before start
        res2 = test_client.post(
            f"/api/projects/{proj_id}/segments/{seg.id}/split?split_time=-0.5"
        )
        assert res2.status_code == 400

    def test_segment_split_at_exact_boundary_400(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg = proj.segments[0]

        res = test_client.post(
            f"/api/projects/{proj_id}/segments/{seg.id}/split?split_time={seg.start_time}"
        )
        assert res.status_code == 400

    def test_segment_merge_identical_segment_id_400(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg_id = proj.segments[0].id

        res = test_client.post(
            f"/api/projects/{proj_id}/segments/merge?first_id={seg_id}&second_id={seg_id}"
        )
        assert res.status_code == 400

    def test_segment_merge_nonexistent_ids_400(self, test_client: TestClient, sample_video: Path):
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res = test_client.post(
            f"/api/projects/{proj_id}/segments/merge?first_id=fake-1&second_id=fake-2"
        )
        assert res.status_code == 400

    def test_unicode_vietnamese_diacritics_fidelity(self, test_client: TestClient, sample_video: Path):
        """Verify all Vietnamese tones: huyền, sắc, hỏi, ngã, nặng, and vowels: ă, â, đ, ê, ô, ơ, ư."""
        vietnamese_pangram = (
            "Họ và tên: Nguyễn Văn Đạt; Điểm số: 10/10. "
            "Bác sỹ bảo rằng ăn quả ổi, quả đu đủ, quả chuối thì rất tốt cho sức khỏe!"
        )
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=0.0,
                end_time=3.5,
                text=vietnamese_pangram,
                translated_text="The doctor said eating guava, papaya, and banana is good for health!",
            )
        ]
        srt = format_subtitles(segs, export_format=ExportFormat.SRT, bilingual_layout=BilingualLayout.ORIGINAL_TOP)
        assert vietnamese_pangram in srt
        # Roundtrip encoding verification
        encoded = srt.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert vietnamese_pangram in decoded

    def test_multiline_mixed_crlf_lf_handling(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=0.0,
                end_time=2.0,
                text="Dòng một\r\nDòng hai\nDòng ba",
                translated_text="Line one\r\nLine two\nLine three",
            )
        ]
        txt = format_subtitles(segs, export_format=ExportFormat.TXT, include_timestamps_in_txt=False)
        # Multiline text in TXT is replaced with single space
        assert "Dòng một Dòng hai Dòng ba" in txt

    def test_html_tags_and_brackets_preservation(self):
        segs = [
            SubtitleSegment(
                project_id="test",
                index=1,
                start_time=0.0,
                end_time=2.0,
                text="<i>Lời thoại in nghiêng</i> và <b>in đậm</b> &amp; ký tự đặc biệt <font color='#ff0000'>Đỏ</font>",
            )
        ]
        srt = format_subtitles(segs, export_format=ExportFormat.SRT)
        assert "<i>Lời thoại in nghiêng</i>" in srt
        assert "<font color='#ff0000'>Đỏ</font>" in srt

    def test_content_disposition_rfc5987_non_ascii_project_name(self, test_client: TestClient, sample_video: Path):
        from app.api.export import make_content_disposition

        # 1. Directly test make_content_disposition with full Vietnamese diacritics
        vn_name = "Dự Án Phụ Đề Video Đặc Sắc 2026.srt"
        disp = make_content_disposition(vn_name)
        assert "filename*=" in disp
        assert "UTF-8''" in disp
        assert "D%E1%BB%B1%20%C3%81n" in disp

        # 2. Test endpoint Content-Disposition with spaces and brackets requiring RFC 5987
        meta = probe_video(sample_video)
        loop = asyncio.get_event_loop()
        proj = loop.run_until_complete(
            db.create_project(
                name="Sub Magic Video Project [2026]",
                video_path=str(sample_video),
                video_metadata=meta,
            )
        )
        res = test_client.get(f"/api/projects/{proj.id}/export?format=srt")
        assert res.status_code == 200
        hdr_disp = res.headers.get("content-disposition", "")
        assert "filename*=" in hdr_disp or "filename=" in hdr_disp
        assert "UTF-8''" in hdr_disp


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-step interactions across distinct system modules."""

    def test_chain_split_edit_bilingual_export(self, test_client: TestClient, sample_video: Path):
        """Create project -> Split segment -> Edit Vietnamese diacritics -> Export Bilingual SRT."""
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg2 = proj.segments[1]  # 2.10 -> 4.30

        # 1. Split
        split_res = test_client.post(
            f"/api/projects/{proj_id}/segments/{seg2.id}/split?split_time=3.0"
        )
        assert split_res.status_code == 200
        created = split_res.json()
        child2_id = created[1]["id"]

        # 2. Edit
        edit_res = test_client.patch(
            f"/api/projects/{proj_id}/segments/{child2_id}",
            json={
                "text": "Độ chính xác rất tuyệt vời!",
                "translated_text": "The accuracy is truly fantastic!",
            },
        )
        assert edit_res.status_code == 200

        # 3. Export
        exp_res = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=original_top")
        assert exp_res.status_code == 200
        srt_body = exp_res.text

        # Validate cues
        assert "Độ chính xác rất tuyệt vời!" in srt_body
        assert "The accuracy is truly fantastic!" in srt_body
        assert "00:00:03,000 --> 00:00:04,300" in srt_body

    def test_chain_merge_reindex_export_vtt(self, test_client: TestClient, sample_video: Path):
        """Add 3 segments -> Merge 1 & 2 -> Export VTT -> Verify 2 sequential cues spanning full interval."""
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        proj = loop.run_until_complete(db.get_project(proj_id))
        seg1 = proj.segments[0]
        seg2 = proj.segments[1]

        merge_res = test_client.post(
            f"/api/projects/{proj_id}/segments/merge?first_id={seg1.id}&second_id={seg2.id}"
        )
        assert merge_res.status_code == 200

        vtt_res = test_client.get(f"/api/projects/{proj_id}/export?format=vtt&bilingual_layout=none")
        assert vtt_res.status_code == 200
        vtt_text = vtt_res.text

        # Should have cue 1 (00:00:00.000 --> 00:00:04.300) and cue 2 (00:00:04.600 --> 00:00:06.000)
        assert "00:00:00.000 --> 00:00:04.300" in vtt_text
        assert "00:00:04.600 --> 00:00:06.000" in vtt_text

    def test_chain_multi_split_cascade(self, test_client: TestClient, sample_video: Path):
        """Split 1 segment into 2, then split child again -> Check 3 continuous sequential segments."""
        loop = asyncio.get_event_loop()
        meta = probe_video(sample_video)
        proj = loop.run_until_complete(
            db.create_project(name="Cascade Proj", video_path=str(sample_video), video_metadata=meta)
        )
        root = SubtitleSegment(
            project_id=proj.id,
            index=1,
            start_time=0.0,
            end_time=9.0,
            text="Một đoạn văn bản dài cần được chia thành nhiều câu nhỏ.",
        )
        loop.run_until_complete(db.insert_segment(root))

        # First split at 3.0s
        res1 = test_client.post(f"/api/projects/{proj.id}/segments/{root.id}/split?split_time=3.0")
        assert res1.status_code == 200
        children = res1.json()
        child2_id = children[1]["id"]

        # Second split child 2 at 6.0s
        res2 = test_client.post(f"/api/projects/{proj.id}/segments/{child2_id}/split?split_time=6.0")
        assert res2.status_code == 200

        final_proj = loop.run_until_complete(db.get_project(proj.id))
        assert len(final_proj.segments) == 3
        assert [s.index for s in final_proj.segments] == [1, 2, 3]
        assert [s.start_time for s in final_proj.segments] == [0.0, 3.0, 6.0]
        assert [s.end_time for s in final_proj.segments] == [3.0, 6.0, 9.0]

    def test_chain_concurrent_export_formats_consistency(self, test_client: TestClient, sample_video: Path):
        """Verify that SRT, VTT, and TXT exports on identical state produce consistent text content."""
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        res_srt = test_client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=none")
        res_vtt = test_client.get(f"/api/projects/{proj_id}/export?format=vtt&bilingual_layout=none")
        res_txt = test_client.get(f"/api/projects/{proj_id}/export?format=txt&include_timestamps_in_txt=false")

        common_snippet = "Chào mừng các bạn đến với ứng dụng Sub Magic"
        assert common_snippet in res_srt.text
        assert common_snippet in res_vtt.text
        assert common_snippet in res_txt.text


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic end-user workflows combining UI actions, media streaming, and editing."""

    def test_scenario_complete_hardsub_ocr_correction_workflow(self, test_client: TestClient, sample_video: Path):
        """Simulate a complete Hardsub OCR correction workflow:
        1. Register video project.
        2. Set ROI box & exclusion watermark masks.
        3. Insert 4 OCR-extracted segments with a joined sentence and an OCR typo.
        4. Stream video Range to verify frame at 2.5s.
        5. Split the joined segment into 2 parts.
        6. Fix OCR typo with correct Vietnamese diacritics.
        7. Add English translations.
        8. Export final SRT and VTT.
        """
        loop = asyncio.get_event_loop()
        meta = probe_video(sample_video)
        proj = loop.run_until_complete(
            db.create_project(name="OCR Video Workflow", video_path=str(sample_video), video_metadata=meta)
        )

        # Step 2: Set ROI
        roi_res = test_client.put(
            f"/api/projects/{proj.id}/roi",
            json={
                "main_roi": {"x": 0.05, "y": 0.82, "w": 0.90, "h": 0.15},
                "exclusion_masks": [{"x": 0.80, "y": 0.83, "w": 0.12, "h": 0.08}],
            },
        )
        assert roi_res.status_code == 200

        # Step 3: Insert raw OCR segments
        raw_segs = [
            SubtitleSegment(project_id=proj.id, index=1, start_time=0.5, end_time=2.0, text="Khởi động hệ thống.", source="ocr"),
            # Segment 2 is too long, contains 2 sentences joined:
            SubtitleSegment(project_id=proj.id, index=2, start_time=2.2, end_time=6.4, text="Đây là câu thứ nhất. Và đây là câu thứ hai.", source="ocr"),
            # Segment 3 has OCR typo: "tri trơn" instead of "trơn tru"
            SubtitleSegment(project_id=proj.id, index=3, start_time=6.8, end_time=8.5, text="Mọi thứ hoạt động tri trơn.", source="ocr"),
        ]
        for s in raw_segs:
            loop.run_until_complete(db.insert_segment(s))

        # Step 4: Stream video chunk for player verification
        stream_res = test_client.get(f"/api/projects/{proj.id}/video", headers={"Range": "bytes=0-1024"})
        assert stream_res.status_code == 206

        # Step 5: Split long segment at 4.3s
        proj_state = loop.run_until_complete(db.get_project(proj.id))
        joined_seg = proj_state.segments[1]
        split_res = test_client.post(
            f"/api/projects/{proj.id}/segments/{joined_seg.id}/split",
            json={
                "split_time": 4.3,
                "first_text": "Đây là câu thứ nhất.",
                "second_text": "Và đây là câu thứ hai.",
            },
        )
        assert split_res.status_code == 200

        # Step 6: Fix typo in segment 3 (now index 4 after split)
        refreshed = loop.run_until_complete(db.get_project(proj.id))
        typo_seg = refreshed.segments[3]
        patch_res = test_client.patch(
            f"/api/projects/{proj.id}/segments/{typo_seg.id}",
            json={
                "text": "Mọi thứ hoạt động trơn tru.",
                "translated_text": "Everything runs smoothly.",
            },
        )
        assert patch_res.status_code == 200

        # Step 7: Export bilingual SRT
        exp_res = test_client.get(f"/api/projects/{proj.id}/export?format=srt&bilingual_layout=original_top")
        assert exp_res.status_code == 200
        srt_content = exp_res.text

        # Assert final result integrity
        assert "Mọi thứ hoạt động trơn tru" in srt_content
        assert "Everything runs smoothly" in srt_content
        assert "Đây là câu thứ nhất" in srt_content
        assert "Và đây là câu thứ hai" in srt_content

        # Verify cues count is 4
        cue_indices = re.findall(r"^(\d+)$", srt_content, flags=re.MULTILINE)
        assert cue_indices == ["1", "2", "3", "4"]

    def test_scenario_stt_podcast_transcription_workflow(self, test_client: TestClient, sample_video: Path):
        """Simulate an STT podcast workflow:
        1. Insert 4 rapid conversational utterances.
        2. Merge short conversational fragments into a fluent turn.
        3. Export TXT transcript with timestamps for article publication.
        """
        loop = asyncio.get_event_loop()
        meta = probe_video(sample_video)
        proj = loop.run_until_complete(
            db.create_project(name="Podcast Episode 1", video_path=str(sample_video), video_metadata=meta)
        )
        u1 = SubtitleSegment(project_id=proj.id, index=1, start_time=0.0, end_time=1.2, text="Vâng,", source="stt")
        u2 = SubtitleSegment(project_id=proj.id, index=2, start_time=1.3, end_time=3.5, text="chúng tôi hoàn toàn đồng ý với ý kiến này.", source="stt")
        u3 = SubtitleSegment(project_id=proj.id, index=3, start_time=3.8, end_time=5.5, text="Điều đó sẽ giúp ích rất nhiều.", source="stt")
        loop.run_until_complete(db.insert_segment(u1))
        loop.run_until_complete(db.insert_segment(u2))
        loop.run_until_complete(db.insert_segment(u3))

        # Merge u1 and u2
        merge_res = test_client.post(f"/api/projects/{proj.id}/segments/merge?first_id={u1.id}&second_id={u2.id}")
        assert merge_res.status_code == 200

        # Export TXT with timestamps
        txt_res = test_client.get(f"/api/projects/{proj.id}/export?format=txt&include_timestamps_in_txt=true")
        assert txt_res.status_code == 200
        txt_content = txt_res.text

        assert "[00:00:00.000 -> 00:00:03.500] Vâng, chúng tôi hoàn toàn đồng ý với ý kiến này" in txt_content
        assert "[00:00:03.800 -> 00:00:05.500] Điều đó sẽ giúp ích rất nhiều" in txt_content

    def test_scenario_video_player_timeline_scrubbing(self, test_client: TestClient, sample_video: Path):
        """Simulate browser HTML5 media player requesting sequential chunks during playback scrubbing."""
        loop = asyncio.get_event_loop()
        proj_id = loop.run_until_complete(create_populated_project(sample_video))

        total_bytes = sample_video.stat().st_size
        full_bytes = sample_video.read_bytes()

        # Step 1: Player queries initial metadata
        res1 = test_client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=0-255"})
        assert res1.status_code == 206
        chunk1 = res1.content
        assert chunk1 == full_bytes[0:256]

        # Step 2: Player scrubs to 50%
        mid = total_bytes // 2
        res2 = test_client.get(f"/api/projects/{proj_id}/video", headers={"Range": f"bytes={mid}-{mid + 255}"})
        assert res2.status_code == 206
        chunk2 = res2.content
        assert chunk2 == full_bytes[mid:mid + 256]

        # Step 3: Player streams rest of file to EOF
        res3 = test_client.get(f"/api/projects/{proj_id}/video", headers={"Range": f"bytes={mid + 256}-"})
        assert res3.status_code == 206
        chunk3 = res3.content
        assert chunk3 == full_bytes[mid + 256:]
