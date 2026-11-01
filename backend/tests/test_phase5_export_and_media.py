"""Comprehensive test suite for Milestone 1 (Phase 5):
- Subtitle Formatter (SRT, VTT, TXT, Bilingual layouts, parse_timestamp, cleaning, wrapping)
- Export API (GET and POST routes, MIME types, Content-Disposition, file saving)
- Video Streaming API (HTTP 200, HTTP 206 Range requests, FileResponse)
- Segment Split & Merge (Database operations, re-indexing, validation, API endpoints)
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import (
    BilingualLayout,
    ExportFormat,
    SubtitleSegment,
    WordTimestamp,
)
from app.services.ffmpeg_service import probe_video
from app.utils.subtitle_formatter import (
    clean_subtitle_text,
    format_subtitles,
    format_timestamp_srt,
    format_timestamp_vtt,
    parse_timestamp,
    parse_timestamp_srt,
    parse_timestamp_vtt,
    wrap_subtitle_lines,
)
from tests.test_phase1_and_phase2 import create_sample_video_with_audio


# ============================================================================
# 1. Subtitle Formatter & Timecode Parser Unit Tests
# ============================================================================

def test_timestamp_formatting_helpers():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(83.456) == "00:01:23,456"
    assert format_timestamp_srt(3665.123) == "01:01:05,123"

    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(83.456) == "00:01:23.456"
    assert format_timestamp_vtt(3665.123) == "01:01:05.123"


def test_parse_timestamp_srt_and_vtt():
    # Standard SRT format (comma)
    assert parse_timestamp("00:01:23,456") == 83.456
    assert parse_timestamp_srt("01:00:00,000") == 3600.0
    assert parse_timestamp("00:00:05,500") == 5.5

    # Standard VTT format (dot)
    assert parse_timestamp("00:01:23.456") == 83.456
    assert parse_timestamp_vtt("01:02:03.500") == 3723.5

    # Short format MM:SS.mmm or MM:SS,mmm
    assert parse_timestamp("01:23.456") == 83.456
    assert parse_timestamp("02:10,250") == 130.25

    # Single seconds
    assert parse_timestamp("45.678") == 45.678
    assert parse_timestamp("120") == 120.0

    # With leading/trailing whitespace
    assert parse_timestamp("  00:00:10,500  ") == 10.5

    # Invalid timestamp format raises ValueError
    with pytest.raises(ValueError):
        parse_timestamp("invalid:time:format:extra")
    with pytest.raises(ValueError):
        parse_timestamp("abc")


def test_clean_and_wrap_subtitle_helpers():
    raw_text = "  Dòng một  \n\n\n  Dòng hai   \n  \n"
    cleaned = clean_subtitle_text(raw_text)
    assert cleaned == "Dòng một\n\nDòng hai"

    empty_cleaned = clean_subtitle_text("")
    assert empty_cleaned == ""

    long_sentence = "Đây là một câu phụ đề tiếng Việt rất dài cần phải được bẻ dòng tự động để vừa khung hình hiển thị"
    wrapped = wrap_subtitle_lines(long_sentence, max_chars=30)
    for line in wrapped.split("\n"):
        assert len(line) <= 35  # Allows boundary without breaking word


def test_format_subtitles_single_and_bilingual():
    segments = [
        SubtitleSegment(
            project_id="test_p",
            index=1,
            start_time=1.0,
            end_time=3.5,
            text="Xin chào các bạn",
            translated_text="Hello everyone",
        ),
        SubtitleSegment(
            project_id="test_p",
            index=2,
            start_time=4.0,
            end_time=6.2,
            text="Chào mừng đến với Sub Magic",
            translated_text="Welcome to Sub Magic",
        ),
    ]

    # SRT - None (Original Only)
    srt_none = format_subtitles(segments, ExportFormat.SRT, BilingualLayout.NONE)
    assert "1\n00:00:01,000 --> 00:00:03,500\nXin chào các bạn" in srt_none
    assert "Hello everyone" not in srt_none

    # SRT - Original Top
    srt_orig_top = format_subtitles(segments, ExportFormat.SRT, BilingualLayout.ORIGINAL_TOP)
    assert "Xin chào các bạn\nHello everyone" in srt_orig_top

    # SRT - Translated Top
    srt_trans_top = format_subtitles(segments, ExportFormat.SRT, BilingualLayout.TRANSLATED_TOP)
    assert "Hello everyone\nXin chào các bạn" in srt_trans_top

    # SRT - Translated Only
    srt_trans_only = format_subtitles(segments, ExportFormat.SRT, BilingualLayout.TRANSLATED_ONLY)
    assert "Hello everyone" in srt_trans_only
    assert "Xin chào các bạn" not in srt_trans_only

    # VTT - Header and format
    vtt_out = format_subtitles(segments, ExportFormat.VTT, BilingualLayout.ORIGINAL_TOP)
    assert vtt_out.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:03.500" in vtt_out
    assert "Xin chào các bạn\nHello everyone" in vtt_out

    # TXT with timestamps
    txt_ts = format_subtitles(
        segments,
        ExportFormat.TXT,
        BilingualLayout.ORIGINAL_TOP,
        include_timestamps_in_txt=True,
    )
    assert "[00:00:01.000 -> 00:00:03.500] Xin chào các bạn | Hello everyone" in txt_ts

    # TXT without timestamps
    txt_plain = format_subtitles(
        segments,
        ExportFormat.TXT,
        BilingualLayout.ORIGINAL_TOP,
        include_timestamps_in_txt=False,
    )
    assert "[00:00:" not in txt_plain
    assert "Xin chào các bạn Hello everyone" in txt_plain


# ============================================================================
# 2. Export API Endpoint Tests (GET & POST, MIME Types, Content-Disposition)
# ============================================================================

@pytest.mark.asyncio
async def test_export_get_and_post_endpoints(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "export_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    project = await db.create_project(
        name="Export Sample Project",
        video_path=str(dummy_video),
    )
    proj_id = project.id

    await db.insert_segment(
        SubtitleSegment(
            project_id=proj_id,
            index=1,
            start_time=0.5,
            end_time=2.5,
            text="Dòng phụ đề 1",
            translated_text="Subtitle line 1",
        )
    )

    client = TestClient(app)

    # 1. GET SRT export
    res_srt = client.get(f"/api/projects/{proj_id}/export?format=srt&bilingual_layout=none")
    assert res_srt.status_code == 200
    assert "application/x-subrip" in res_srt.headers["content-type"]
    assert 'attachment; filename="Export Sample Project.srt"' in res_srt.headers["content-disposition"]
    assert "00:00:00,500 --> 00:00:02,500" in res_srt.text
    assert "Dòng phụ đề 1" in res_srt.text
    assert "Subtitle line 1" not in res_srt.text

    # 2. GET VTT bilingual export
    res_vtt = client.get(f"/api/projects/{proj_id}/export?format=vtt&bilingual_layout=translated_top")
    assert res_vtt.status_code == 200
    assert "text/vtt" in res_vtt.headers["content-type"]
    assert 'attachment; filename="Export Sample Project.vtt"' in res_vtt.headers["content-disposition"]
    assert "WEBVTT" in res_vtt.text
    assert "Subtitle line 1\nDòng phụ đề 1" in res_vtt.text

    # 3. GET TXT export with timestamps
    res_txt_ts = client.get(
        f"/api/projects/{proj_id}/export?format=txt&bilingual_layout=original_top&include_timestamps_in_txt=true"
    )
    assert res_txt_ts.status_code == 200
    assert "text/plain" in res_txt_ts.headers["content-type"]
    assert "[00:00:00.500 -> 00:00:02.500]" in res_txt_ts.text

    # 4. GET TXT export without timestamps
    res_txt_plain = client.get(
        f"/api/projects/{proj_id}/export?format=txt&bilingual_layout=none&include_timestamps_in_txt=false"
    )
    assert res_txt_plain.status_code == 200
    assert "[00:00:" not in res_txt_plain.text
    assert "Dòng phụ đề 1" in res_txt_plain.text

    # 5. POST export with custom save_to_disk_path
    custom_out = tmp_path / "custom_dir" / "my_custom_export.srt"
    res_post = client.post(
        f"/api/projects/{proj_id}/export",
        json={
            "format": "srt",
            "bilingual_layout": "original_top",
            "save_to_disk_path": str(custom_out),
        },
    )
    assert res_post.status_code == 200
    assert custom_out.exists()
    assert "Dòng phụ đề 1\nSubtitle line 1" in custom_out.read_text(encoding="utf-8")
    assert "X-Saved-Path" in res_post.headers

    # 6. 404 for nonexistent project
    res_404 = client.get("/api/projects/nonexistent-id/export")
    assert res_404.status_code == 404


# ============================================================================
# 3. Video Streaming & HTTP 206 Range Request Tests
# ============================================================================

@pytest.mark.asyncio
async def test_video_streaming_and_range_requests(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "streaming_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    video_path = create_sample_video_with_audio(tmp_path / "stream_video.mp4", duration_sec=2.0)
    meta = probe_video(video_path)
    file_size = video_path.stat().st_size
    assert file_size > 1000

    project = await db.create_project(
        name="Stream Test Project",
        video_path=str(video_path),
        video_metadata=meta,
    )
    proj_id = project.id

    client = TestClient(app)

    # 1. Full GET without Range header -> 200 OK
    res_full = client.get(f"/api/projects/{proj_id}/video")
    assert res_full.status_code == 200
    assert res_full.headers.get("accept-ranges") == "bytes"
    assert res_full.headers.get("content-type") == "video/mp4"
    assert len(res_full.content) == file_size

    # 2. GET with Range: bytes=0-499 -> 206 Partial Content
    res_range1 = client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=0-499"})
    assert res_range1.status_code == 206
    assert len(res_range1.content) == 500
    assert res_range1.headers.get("content-range") == f"bytes 0-499/{file_size}"
    assert res_range1.headers.get("content-length") == "500"

    # 3. GET with Range: bytes=500-999 -> 206 Partial Content
    res_range2 = client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=500-999"})
    assert res_range2.status_code == 206
    assert len(res_range2.content) == 500
    assert res_range2.headers.get("content-range") == f"bytes 500-999/{file_size}"

    # 4. GET with Range: bytes=1000- -> 206 Partial Content to end
    res_range3 = client.get(f"/api/projects/{proj_id}/video", headers={"Range": "bytes=1000-"})
    assert res_range3.status_code == 206
    expected_len = file_size - 1000
    assert len(res_range3.content) == expected_len
    assert res_range3.headers.get("content-range") == f"bytes 1000-{file_size - 1}/{file_size}"

    # 5. Nonexistent project -> 404
    res_notfound = client.get("/api/projects/unknown-proj/video")
    assert res_notfound.status_code == 404


# ============================================================================
# 4. Segment Split & Merge Tests (DB & API)
# ============================================================================

@pytest.mark.asyncio
async def test_segment_split_and_merge_database_and_api(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "split_merge_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy_sm.mp4"
    dummy_video.write_bytes(b"dummy")

    project = await db.create_project(
        name="Split Merge Project",
        video_path=str(dummy_video),
    )
    proj_id = project.id

    # Create 3 segments
    s1 = await db.insert_segment(
        SubtitleSegment(
            project_id=proj_id,
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="Đoạn thứ nhất",
        )
    )
    s2 = await db.insert_segment(
        SubtitleSegment(
            project_id=proj_id,
            index=2,
            start_time=2.5,
            end_time=6.0,
            text="Đoạn thứ hai phần đầu và phần sau",
            words=[
                WordTimestamp(word="Đoạn", start=2.5, end=3.0),
                WordTimestamp(word="thứ", start=3.0, end=3.5),
                WordTimestamp(word="hai", start=3.5, end=4.0),
                WordTimestamp(word="phần", start=4.1, end=4.6),
                WordTimestamp(word="sau", start=4.6, end=5.8),
            ],
        )
    )
    s3 = await db.insert_segment(
        SubtitleSegment(
            project_id=proj_id,
            index=3,
            start_time=7.0,
            end_time=9.0,
            text="Đoạn thứ ba",
        )
    )

    # 1. DB Split s2 at 4.0
    split_res = await db.split_segment(
        project_id=proj_id,
        segment_id=s2.id,
        split_time=4.0,
        first_text="Đoạn thứ hai phần đầu",
        second_text="và phần sau",
    )
    assert len(split_res) == 2
    first_part, second_part = split_res[0], split_res[1]

    assert first_part.id == s2.id
    assert first_part.start_time == 2.5
    assert first_part.end_time == 4.0
    assert first_part.text == "Đoạn thứ hai phần đầu"
    assert first_part.index == 2

    assert second_part.id != s2.id
    assert second_part.start_time == 4.0
    assert second_part.end_time == 6.0
    assert second_part.text == "và phần sau"
    assert second_part.index == 3

    # Check that s3 has been reindexed to index 4
    proj_detail = await db.get_project(proj_id)
    assert len(proj_detail.segments) == 4
    assert proj_detail.segments[0].index == 1
    assert proj_detail.segments[1].index == 2
    assert proj_detail.segments[2].index == 3
    assert proj_detail.segments[3].index == 4
    assert proj_detail.segments[3].text == "Đoạn thứ ba"

    # 2. Test invalid split_time raises ValueError
    with pytest.raises(ValueError):
        await db.split_segment(proj_id, first_part.id, split_time=1.0)  # before start_time
    with pytest.raises(ValueError):
        await db.split_segment(proj_id, first_part.id, split_time=5.0)  # after end_time

    # 3. Test API split endpoint via JSON body
    client = TestClient(app)
    api_split_res = client.post(
        f"/api/projects/{proj_id}/segments/{first_part.id}/split",
        json={"split_time": 3.2, "first_text": "Đoạn thứ", "second_text": "hai phần đầu"},
    )
    assert api_split_res.status_code == 200
    res_segs = api_split_res.json()
    assert len(res_segs) == 2
    assert res_segs[0]["text"] == "Đoạn thứ"
    assert res_segs[1]["text"] == "hai phần đầu"

    # 4. Test API split endpoint via query params
    api_split_qp = client.post(
        f"/api/projects/{proj_id}/segments/{s1.id}/split?split_time=1.0"
    )
    assert api_split_qp.status_code == 200
    assert len(api_split_qp.json()) == 2

    # 5. Test API merge endpoint
    merge_res = client.post(
        f"/api/projects/{proj_id}/segments/merge?first_id={s1.id}&second_id={second_part.id}"
    )
    assert merge_res.status_code == 200
    merged_data = merge_res.json()
    assert merged_data["start_time"] == min(s1.start_time, second_part.start_time)


@pytest.mark.asyncio
async def test_export_filename_sanitization_and_vietnamese_headers(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "export_sanitization.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy_vid.mp4"
    dummy_video.write_bytes(b"dummy")

    client = TestClient(app)

    # 1. Windows forbidden characters sanitization
    forbidden_title = 'Sub: Magic * "Star" <Test> ? | / \\'
    proj_forbidden = await db.create_project(name=forbidden_title, video_path=str(dummy_video))
    await db.insert_segment(
        SubtitleSegment(project_id=proj_forbidden.id, index=1, start_time=0.0, end_time=2.0, text="Sanitization test")
    )

    res_forbidden = client.get(f"/api/projects/{proj_forbidden.id}/export?format=srt")
    assert res_forbidden.status_code == 200
    assert "Sanitization test" in res_forbidden.text
    # Header X-Saved-Path must exist and not contain raw forbidden characters
    saved_path_hdr = res_forbidden.headers.get("x-saved-path", "")
    assert saved_path_hdr != ""
    filename_part = Path(saved_path_hdr).name
    assert not any(c in filename_part for c in ['*', '?', ':', '"', '<', '>', '|', '/', '\\'])

    # 2. Vietnamese unicode characters & Latin-1 HTTP header safety
    vietnamese_title = "Dự Án Phụ Đề Tiếng Việt Cực Chuẩn (2026)"
    proj_vn = await db.create_project(name=vietnamese_title, video_path=str(dummy_video))
    await db.insert_segment(
        SubtitleSegment(project_id=proj_vn.id, index=1, start_time=0.0, end_time=2.0, text="Tiếng Việt có dấu")
    )

    res_vn_get = client.get(f"/api/projects/{proj_vn.id}/export?format=vtt")
    assert res_vn_get.status_code == 200
    assert "Tiếng Việt có dấu" in res_vn_get.text

    # X-Saved-Path must be strictly Latin-1 encodable (never crash Starlette)
    vn_saved_path = res_vn_get.headers.get("x-saved-path", "")
    assert vn_saved_path != ""
    vn_saved_path.encode("latin-1")  # Raises UnicodeEncodeError if not Latin-1 safe

    # 3. POST export with Vietnamese custom path
    vn_custom_dir = tmp_path / "thư_mục_xuất_phụ_đề"
    vn_custom_file = vn_custom_dir / "kết_quả.srt"
    res_vn_post = client.post(
        f"/api/projects/{proj_vn.id}/export",
        json={
            "format": "srt",
            "bilingual_layout": "none",
            "save_to_disk_path": str(vn_custom_file),
        },
    )
    assert res_vn_post.status_code == 200
    assert vn_custom_file.exists()
    post_saved_path = res_vn_post.headers.get("x-saved-path", "")
    post_saved_path.encode("latin-1")
