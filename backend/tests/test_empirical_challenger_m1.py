"""Empirical Challenger Stress & Edge-Case Suite for Milestone 1:
- Subtitle Formatter Tricky Inputs: empty segment list, empty texts, multiline, CRLF, Vietnamese diacritics, HTML/markup, timestamp boundaries.
- Video Range Requests: single-byte (0-0, intermediate, last-byte), suffix ranges (-500, -1), out-of-bound ranges (416), malformed ranges, deleted video file on disk (404).
- Export API Query Combinations: all 24 permutations of format x bilingual_layout x include_timestamps, invalid query parameters (422), non-existent project (404), special characters in project name.
- Split Segment Boundary & Cascade Stress: exact boundary rejection, word timestamps partition fidelity, index resequencing stress.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from urllib.parse import unquote
import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import (
    BilingualLayout,
    ExportFormat,
    ExportRequest,
    SubtitleSegment,
    WordTimestamp,
)
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
# Section 1: Subtitle Formatter Tricky Inputs
# ============================================================================

class TestSubtitleFormatterTrickyInputs:

    def test_empty_segment_list_and_empty_text(self):
        """Empty lists or segments with blank text should yield clean empty/header outputs."""
        # 1. Completely empty segments list
        assert format_subtitles([], ExportFormat.SRT) == ""
        assert format_subtitles([], ExportFormat.VTT) == "WEBVTT\n"
        assert format_subtitles([], ExportFormat.TXT) == ""

        # 2. Segments containing only whitespace or None
        blank_segments = [
            SubtitleSegment(project_id="p", index=1, start_time=1.0, end_time=2.0, text="   ", translated_text=None),
            SubtitleSegment(project_id="p", index=2, start_time=3.0, end_time=4.0, text="", translated_text="  \n  "),
            SubtitleSegment(project_id="p", index=3, start_time=5.0, end_time=6.0, text="\t\r\n", translated_text=""),
        ]
        assert format_subtitles(blank_segments, ExportFormat.SRT) == ""
        assert format_subtitles(blank_segments, ExportFormat.VTT) == "WEBVTT\n"
        assert format_subtitles(blank_segments, ExportFormat.TXT) == ""

    def test_multiline_text_and_mixed_crlf_preservation(self):
        """Subtitle blocks with multiline and Windows CRLF should format cleanly without corrupting cues."""
        raw_text = "Dòng 1\r\nDòng 2\r\n\r\n\r\nDòng 3"
        cleaned = clean_subtitle_text(raw_text)
        # Should collapse consecutive blank lines into at most one blank line
        assert cleaned == "Dòng 1\nDòng 2\n\nDòng 3"

        seg = SubtitleSegment(
            project_id="p",
            index=1,
            start_time=1.0,
            end_time=4.0,
            text="Dòng thứ nhất\r\nDòng thứ hai",
            translated_text="First line\r\nSecond line",
        )

        # SRT Original Top
        srt_out = format_subtitles([seg], ExportFormat.SRT, bilingual_layout=BilingualLayout.ORIGINAL_TOP)
        expected_body = "Dòng thứ nhất\nDòng thứ hai\nFirst line\nSecond line"
        assert f"1\n00:00:01,000 --> 00:00:04,000\n{expected_body}\n" == srt_out

        # TXT with timestamps (newlines should become " | ")
        txt_ts = format_subtitles([seg], ExportFormat.TXT, bilingual_layout=BilingualLayout.ORIGINAL_TOP, include_timestamps_in_txt=True)
        assert "[00:00:01.000 -> 00:00:04.000] Dòng thứ nhất | Dòng thứ hai | First line | Second line\n" == txt_ts

        # TXT without timestamps (newlines should become spaces)
        txt_no_ts = format_subtitles([seg], ExportFormat.TXT, bilingual_layout=BilingualLayout.ORIGINAL_TOP, include_timestamps_in_txt=False)
        assert "Dòng thứ nhất Dòng thứ hai First line Second line\n" == txt_no_ts

    def test_vietnamese_diacritics_and_unicode_stress(self):
        """Vietnamese full tone marks and special Unicode symbols must round-trip with 100% fidelity."""
        vietnamese_tone_sample = (
            "Tiếng Việt giàu đẹp: ắ, ằ, ẳ, ẵ, ặ, ế, ề, ể, ễ, ệ, ố, ồ, ổ, ỗ, ộ, "
            "ư, ứ, ừ, ử, ữ, ự, ý, ỳ, ỷ, ỹ, ỵ, đ, Đ. 🎬🚀 100% “chính xác” — test!"
        )
        seg = SubtitleSegment(
            project_id="p",
            index=1,
            start_time=0.5,
            end_time=3.5,
            text=vietnamese_tone_sample,
            translated_text="Vietnamese language rich and beautiful: 100% accuracy!",
        )

        # Ensure wrapping does not split characters or corrupt tone marks
        wrapped = wrap_subtitle_lines(vietnamese_tone_sample, max_chars=40)
        assert "tiếng" in wrapped.lower()
        assert "giàu đẹp" in wrapped
        assert "🎬🚀" in wrapped

        # Format as SRT, VTT, and TXT
        for fmt in [ExportFormat.SRT, ExportFormat.VTT, ExportFormat.TXT]:
            out = format_subtitles([seg], export_format=fmt, bilingual_layout=BilingualLayout.ORIGINAL_TOP)
            assert "Tiếng Việt giàu đẹp" in out
            assert "ắ, ằ, ẳ, ẵ, ặ" in out
            assert "🎬🚀" in out
            assert "“chính xác”" in out

    def test_html_tags_and_brackets_preservation(self):
        """HTML formatting tags (<i>, <b>, <font>, <c.color>) and < > math symbols must not be corrupted."""
        markup_text = "<i>Đoạn nghiêng</i> và <b>in đậm</b> cùng <font color='#ffff00'>màu vàng</font>"
        math_text = "Điều kiện: a < 10 & b > 20 (C++ && Python)"
        seg1 = SubtitleSegment(project_id="p", index=1, start_time=1.0, end_time=2.5, text=markup_text)
        seg2 = SubtitleSegment(project_id="p", index=2, start_time=3.0, end_time=5.0, text=math_text)

        srt_out = format_subtitles([seg1, seg2], ExportFormat.SRT)
        assert "<i>Đoạn nghiêng</i>" in srt_out
        assert "<b>in đậm</b>" in srt_out
        assert "<font color='#ffff00'>màu vàng</font>" in srt_out
        assert "a < 10 & b > 20 (C++ && Python)" in srt_out

        vtt_out = format_subtitles([seg1, seg2], ExportFormat.VTT)
        assert "<i>Đoạn nghiêng</i>" in vtt_out
        assert "a < 10 & b > 20" in vtt_out

    def test_timestamp_boundary_values(self):
        """Test boundary conditions for timestamps: zero, sub-millisecond, inverted, negative, high hours."""
        # 1. Zero seconds
        assert format_timestamp_srt(0.0) == "00:00:00,000"
        assert format_timestamp_vtt(0.0) == "00:00:00.000"

        # 2. Sub-millisecond rounding
        assert format_timestamp_srt(0.0004) == "00:00:00,000"
        assert format_timestamp_srt(0.0006) == "00:00:00,001"
        assert format_timestamp_srt(12.3456) == "00:00:12,346"

        # 3. High hour values
        assert format_timestamp_srt(366100.123) == "101:41:40,123"
        assert format_timestamp_vtt(366100.123) == "101:41:40.123"

        # 4. Inverted / zero duration clamp in format_subtitles
        inverted_seg = SubtitleSegment(
            project_id="p",
            index=1,
            start_time=10.0,
            end_time=5.0,  # end before start
            text="Inverted time segment",
        )
        srt = format_subtitles([inverted_seg], ExportFormat.SRT)
        # End time should be clamped to start_time + 0.1
        assert "00:00:10,000 --> 00:00:10,100" in srt

        # 5. Negative timestamp clamp
        assert format_timestamp_srt(-5.0) == "00:00:00,000"
        assert format_timestamp_vtt(-5.0) == "00:00:00.000"

    def test_parse_timestamp_comprehensive(self):
        """Test parse_timestamp parser with various valid and invalid formats."""
        # Valid cases
        assert parse_timestamp("00:00:00,000") == 0.0
        assert parse_timestamp("00:00:00.000") == 0.0
        assert parse_timestamp("01:23:45,678") == 5025.678
        assert parse_timestamp("01:23:45.678") == 5025.678
        assert parse_timestamp("05:30.500") == 330.5
        assert parse_timestamp("42.5") == 42.5
        assert parse_timestamp("100") == 100.0
        assert parse_timestamp_srt("00:01:00,000") == 60.0
        assert parse_timestamp_vtt("00:02:00.000") == 120.0

        # Invalid cases raise ValueError
        with pytest.raises(ValueError):
            parse_timestamp("")
        with pytest.raises(ValueError):
            parse_timestamp("   ")
        with pytest.raises(ValueError):
            parse_timestamp("not:a:time")
        with pytest.raises(ValueError):
            parse_timestamp("01:02:03:04:05")


# ============================================================================
# Section 2: Video Range Requests Empirical Tests
# ============================================================================

@pytest.fixture
async def video_project_fixture(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "range_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db)
    await db.init_db()

    video_file = create_sample_video_with_audio(tmp_path / "sample_video.mp4", duration_sec=1.5)
    project = await db.create_project(name="Video Test Project", video_path=str(video_file))
    return project, video_file


@pytest.mark.asyncio
class TestVideoRangeRequests:

    async def test_single_byte_ranges(self, video_project_fixture):
        """Empirical verification of single-byte Range requests: 0-0, intermediate, last-byte."""
        project, video_file = video_project_fixture
        video_bytes = video_file.read_bytes()
        file_size = len(video_bytes)
        assert file_size > 1000

        client = TestClient(app)

        # 1. First byte (bytes=0-0)
        res_first = client.get(f"/api/projects/{project.id}/video", headers={"Range": "bytes=0-0"})
        assert res_first.status_code == 206
        assert len(res_first.content) == 1
        assert res_first.content == video_bytes[0:1]
        assert res_first.headers["content-range"] == f"bytes 0-0/{file_size}"
        assert res_first.headers["content-length"] == "1"

        # 2. Intermediate byte (bytes=50-50)
        res_mid = client.get(f"/api/projects/{project.id}/video", headers={"Range": "bytes=50-50"})
        assert res_mid.status_code == 206
        assert len(res_mid.content) == 1
        assert res_mid.content == video_bytes[50:51]
        assert res_mid.headers["content-range"] == f"bytes 50-50/{file_size}"
        assert res_mid.headers["content-length"] == "1"

        # 3. Last byte (bytes={size-1}-{size-1})
        last_idx = file_size - 1
        res_last = client.get(f"/api/projects/{project.id}/video", headers={"Range": f"bytes={last_idx}-{last_idx}"})
        assert res_last.status_code == 206
        assert len(res_last.content) == 1
        assert res_last.content == video_bytes[-1:]
        assert res_last.headers["content-range"] == f"bytes {last_idx}-{last_idx}/{file_size}"
        assert res_last.headers["content-length"] == "1"

    async def test_suffix_ranges(self, video_project_fixture):
        """Empirical verification of suffix Range requests: bytes=-500 and bytes=-1."""
        project, video_file = video_project_fixture
        video_bytes = video_file.read_bytes()
        file_size = len(video_bytes)

        client = TestClient(app)

        # 1. Suffix range -500 (last 500 bytes)
        res_suffix500 = client.get(f"/api/projects/{project.id}/video", headers={"Range": "bytes=-500"})
        assert res_suffix500.status_code == 206
        assert len(res_suffix500.content) == 500
        assert res_suffix500.content == video_bytes[-500:]
        assert res_suffix500.headers["content-range"] == f"bytes {file_size - 500}-{file_size - 1}/{file_size}"
        assert res_suffix500.headers["content-length"] == "500"

        # 2. Suffix range -1 (very last byte)
        res_suffix1 = client.get(f"/api/projects/{project.id}/video", headers={"Range": "bytes=-1"})
        assert res_suffix1.status_code == 206
        assert len(res_suffix1.content) == 1
        assert res_suffix1.content == video_bytes[-1:]
        assert res_suffix1.headers["content-range"] == f"bytes {file_size - 1}-{file_size - 1}/{file_size}"

    async def test_out_of_bound_and_malformed_ranges(self, video_project_fixture):
        """Verify HTTP 416 Range Not Satisfiable for out-of-bound ranges and fallback for malformed headers."""
        project, video_file = video_project_fixture
        file_size = video_file.stat().st_size

        client = TestClient(app)

        # 1. Out-of-bound range beyond total file size
        res_oob = client.get(
            f"/api/projects/{project.id}/video",
            headers={"Range": f"bytes={file_size + 1000}-{file_size + 2000}"},
        )
        assert res_oob.status_code == 416
        assert res_oob.headers.get("content-range") == f"bytes */{file_size}"

        # 2. Inverted range where start > end
        res_inverted = client.get(
            f"/api/projects/{project.id}/video",
            headers={"Range": "bytes=300-100"},
        )
        assert res_inverted.status_code in (400, 416)

        # 3. Malformed range header: RFC 7233 / Starlette returns 400 Bad Request or 200/416
        res_malformed = client.get(
            f"/api/projects/{project.id}/video",
            headers={"Range": "bytes=abc-xyz"},
        )
        assert res_malformed.status_code in (200, 400, 416)

    async def test_deleted_video_file_on_disk(self, video_project_fixture):
        """If project exists in database but file is deleted from disk, return 404."""
        project, video_file = video_project_fixture
        video_file.unlink()  # Delete from disk
        assert not video_file.exists()

        client = TestClient(app)
        res = client.get(f"/api/projects/{project.id}/video")
        assert res.status_code == 404
        assert "Không tìm thấy file video" in res.json()["detail"]


# ============================================================================
# Section 3: Export Endpoints Query Combinations & Permutations
# ============================================================================

@pytest.mark.asyncio
class TestExportEndpointsPermutations:

    async def test_all_24_export_query_permutations(self, tmp_path: Path, monkeypatch):
        """Permute format (3) x bilingual_layout (4) x include_timestamps (2) = 24 permutations."""
        test_db = tmp_path / "export_perm_test.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy_data")

        project = await db.create_project(name="Permutation Project", video_path=str(dummy_vid))
        await db.insert_segment(
            SubtitleSegment(
                project_id=project.id,
                index=1,
                start_time=1.0,
                end_time=3.0,
                text="Xin chào",
                translated_text="Hello",
            )
        )

        client = TestClient(app)

        formats = [ExportFormat.SRT, ExportFormat.VTT, ExportFormat.TXT]
        layouts = [
            BilingualLayout.NONE,
            BilingualLayout.ORIGINAL_TOP,
            BilingualLayout.TRANSLATED_TOP,
            BilingualLayout.TRANSLATED_ONLY,
        ]
        timestamps = [True, False]

        tested_count = 0
        for fmt, layout, incl_ts in itertools.product(formats, layouts, timestamps):
            res = client.get(
                f"/api/projects/{project.id}/export",
                params={
                    "format": fmt.value,
                    "bilingual_layout": layout.value,
                    "include_timestamps_in_txt": str(incl_ts).lower(),
                },
            )
            assert res.status_code == 200, f"Failed for {fmt}, {layout}, {incl_ts}"
            # Check content-disposition
            assert "Content-Disposition" in res.headers
            assert f".{fmt.value}" in res.headers["Content-Disposition"]

            # Check MIME type
            if fmt == ExportFormat.SRT:
                assert "application/x-subrip" in res.headers["content-type"]
            elif fmt == ExportFormat.VTT:
                assert "text/vtt" in res.headers["content-type"]
            elif fmt == ExportFormat.TXT:
                assert "text/plain" in res.headers["content-type"]

            # Check content consistency
            text = res.text
            if layout == BilingualLayout.NONE:
                assert "Xin chào" in text
            elif layout == BilingualLayout.TRANSLATED_ONLY:
                assert "Hello" in text
            elif layout in (BilingualLayout.ORIGINAL_TOP, BilingualLayout.TRANSLATED_TOP):
                assert "Xin chào" in text
                assert "Hello" in text

            tested_count += 1

        assert tested_count == 24

    async def test_invalid_query_parameters_validation(self, tmp_path: Path, monkeypatch):
        """Invalid format or bilingual layout query parameters should return 422 Unprocessable Entity."""
        test_db = tmp_path / "invalid_query_test.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy")
        project = await db.create_project(name="P", video_path=str(dummy_vid))

        client = TestClient(app)

        # 1. Invalid format
        res_bad_fmt = client.get(f"/api/projects/{project.id}/export?format=docx")
        assert res_bad_fmt.status_code == 422

        # 2. Invalid bilingual layout
        res_bad_layout = client.get(f"/api/projects/{project.id}/export?bilingual_layout=reversed_side")
        assert res_bad_layout.status_code == 422

        # 3. 404 for non-existent project
        res_404 = client.get("/api/projects/unknown-uuid-9999/export")
        assert res_404.status_code == 404

    async def test_project_with_special_characters_filename_rfc5987(self, tmp_path: Path, monkeypatch):
        """Exporting a project with complex Vietnamese unicode name must produce valid Content-Disposition without crashing."""
        test_db = tmp_path / "spec_chars_test.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy")

        special_name = "Dự án Tiếng Việt 100% — Khóa học #1 (2026)"
        project = await db.create_project(name=special_name, video_path=str(dummy_vid))
        await db.insert_segment(SubtitleSegment(project_id=project.id, index=1, start_time=0.0, end_time=2.0, text="Nội dung"))

        client = TestClient(app)
        res = client.get(f"/api/projects/{project.id}/export?format=srt")
        assert res.status_code == 200

        cd = res.headers["content-disposition"]
        # Must contain RFC 5987 UTF-8 encoded filename
        assert "filename*=" in cd or "filename=" in cd
        assert "UTF-8''" in cd
        # Verify decoding matches original name
        encoded_part = cd.split("UTF-8''")[-1].strip('"')
        assert special_name in unquote(encoded_part)

    async def test_post_export_with_vietnamese_disk_path(self, tmp_path: Path, monkeypatch):
        """Exporting with a custom save_to_disk_path containing Vietnamese characters must not crash with UnicodeEncodeError."""
        test_db = tmp_path / "post_spec_chars_test.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy")
        project = await db.create_project(name="AsciiProject", video_path=str(dummy_vid))
        await db.insert_segment(SubtitleSegment(project_id=project.id, index=1, start_time=0.0, end_time=2.0, text="Nội dung"))

        vn_out_path = tmp_path / "thư_mục_xuất" / "phụ_đề.srt"
        client = TestClient(app)
        res = client.post(
            f"/api/projects/{project.id}/export",
            json={
                "format": "srt",
                "bilingual_layout": "none",
                "save_to_disk_path": str(vn_out_path),
            },
        )
        assert res.status_code == 200
        assert vn_out_path.exists()



# ============================================================================
# Section 4: Segment Split & Database Integrity Stress
# ============================================================================

@pytest.mark.asyncio
class TestSegmentSplitDatabaseStress:

    async def test_segment_split_boundary_rejection(self, tmp_path: Path, monkeypatch):
        """Attempting to split at exactly start_time or end_time or outside boundary must fail with 400."""
        test_db = tmp_path / "split_bound_test.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy")
        project = await db.create_project(name="Split Bound Project", video_path=str(dummy_vid))

        seg = await db.insert_segment(
            SubtitleSegment(project_id=project.id, index=1, start_time=2.0, end_time=6.0, text="Đoạn văn cần tách")
        )

        client = TestClient(app)

        # 1. Split at exact start_time (2.0) -> 400
        res_at_start = client.post(
            f"/api/projects/{project.id}/segments/{seg.id}/split",
            json={"split_time": 2.0},
        )
        assert res_at_start.status_code == 400
        assert "between start_time" in res_at_start.json()["detail"]

        # 2. Split at exact end_time (6.0) -> 400
        res_at_end = client.post(
            f"/api/projects/{project.id}/segments/{seg.id}/split",
            json={"split_time": 6.0},
        )
        assert res_at_end.status_code == 400

        # 3. Split before start_time (1.0) -> 400
        res_before = client.post(
            f"/api/projects/{project.id}/segments/{seg.id}/split",
            json={"split_time": 1.0},
        )
        assert res_before.status_code == 400

        # 4. Split after end_time (7.0) -> 400
        res_after = client.post(
            f"/api/projects/{project.id}/segments/{seg.id}/split",
            json={"split_time": 7.0},
        )
        assert res_after.status_code == 400

    async def test_cascade_split_and_sequential_index_integrity(self, tmp_path: Path, monkeypatch):
        """Performing multiple cascade splits must preserve strict 1..N index sequence and zero gap."""
        test_db = tmp_path / "cascade_split.sqlite3"
        monkeypatch.setattr(db, "db_path", test_db)
        await db.init_db()

        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy")
        project = await db.create_project(name="Cascade Project", video_path=str(dummy_vid))

        # Insert 3 original segments
        s1 = await db.insert_segment(SubtitleSegment(project_id=project.id, index=1, start_time=0.0, end_time=10.0, text="S1"))
        s2 = await db.insert_segment(SubtitleSegment(project_id=project.id, index=2, start_time=10.0, end_time=20.0, text="S2"))
        s3 = await db.insert_segment(SubtitleSegment(project_id=project.id, index=3, start_time=20.0, end_time=30.0, text="S3"))

        client = TestClient(app)

        # Split s1 at 5.0 -> returns [s1_first, s1_second]
        res1 = client.post(f"/api/projects/{project.id}/segments/{s1.id}/split", json={"split_time": 5.0})
        assert res1.status_code == 200
        split_pair1 = res1.json()
        assert len(split_pair1) == 2
        assert split_pair1[0]["index"] == 1
        assert split_pair1[1]["index"] == 2

        # Check project has 4 segments total: [1, 2, 3, 4]
        res_proj1 = client.get(f"/api/projects/{project.id}")
        assert res_proj1.status_code == 200
        all_segs1 = res_proj1.json()["segments"]
        assert len(all_segs1) == 4
        assert [s["index"] for s in all_segs1] == [1, 2, 3, 4]

        # Find the new segment at index 2 and split it at 7.5
        s1b_id = next(s["id"] for s in all_segs1 if s["index"] == 2)
        res2 = client.post(f"/api/projects/{project.id}/segments/{s1b_id}/split", json={"split_time": 7.5})
        assert res2.status_code == 200

        # Check project has 5 segments total: [1, 2, 3, 4, 5]
        res_proj2 = client.get(f"/api/projects/{project.id}")
        all_segs2 = res_proj2.json()["segments"]
        assert len(all_segs2) == 5
        assert [s["index"] for s in all_segs2] == [1, 2, 3, 4, 5]

        # Find the segment at index 5 (original s3) and split it at 25.0 -> total becomes 6 segments
        s3_id = next(s["id"] for s in all_segs2 if s["index"] == 5)
        res3 = client.post(f"/api/projects/{project.id}/segments/{s3_id}/split", json={"split_time": 25.0})
        assert res3.status_code == 200

        # Verify database persisted state matches
        db_proj = await db.get_project(project.id)
        assert len(db_proj.segments) == 6
        assert [s.index for s in db_proj.segments] == [1, 2, 3, 4, 5, 6]
        # Timestamps must be monotonically strictly non-decreasing
        for i in range(len(db_proj.segments) - 1):
            assert db_proj.segments[i].end_time <= db_proj.segments[i + 1].start_time
