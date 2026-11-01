"""Challenger Stress Test Suite for Milestone 1:
- Segment Split & Merge Logic (Word timestamps, contiguous indexes, boundary values)
- Boundary cases: exact start/end time, submillisecond, single word, empty words
- Non-existent segments & invalid inputs
- Sequential split cascades & index integrity
"""
from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.database import db
from app.main import app
from app.schemas import SubtitleSegment, WordTimestamp


# ============================================================================
# 1. Boundary Cases: Exact Start/End Time and Non-Existent Segments
# ============================================================================

@pytest.mark.asyncio
async def test_split_boundary_exact_start_and_end(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_boundary_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Boundary Proj", video_path=str(dummy_video))
    seg = await db.insert_segment(
        SubtitleSegment(
            project_id=project.id,
            index=1,
            start_time=2.0,
            end_time=5.0,
            text="Boundary segment",
        )
    )

    # 1. Split at exact start_time (2.0) -> must raise ValueError
    with pytest.raises(ValueError, match="strictly between"):
        await db.split_segment(project.id, seg.id, split_time=2.0)

    # 2. Split at exact end_time (5.0) -> must raise ValueError
    with pytest.raises(ValueError, match="strictly between"):
        await db.split_segment(project.id, seg.id, split_time=5.0)

    # 3. Split before start_time (1.5) -> must raise ValueError
    with pytest.raises(ValueError, match="strictly between"):
        await db.split_segment(project.id, seg.id, split_time=1.5)

    # 4. Split after end_time (5.5) -> must raise ValueError
    with pytest.raises(ValueError, match="strictly between"):
        await db.split_segment(project.id, seg.id, split_time=5.5)

    # 5. API calls for boundaries -> must return HTTP 400
    client = TestClient(app)
    res_start = client.post(f"/api/projects/{project.id}/segments/{seg.id}/split?split_time=2.0")
    assert res_start.status_code == 400
    assert "strictly between" in res_start.json()["detail"]

    res_end = client.post(f"/api/projects/{project.id}/segments/{seg.id}/split?split_time=5.0")
    assert res_end.status_code == 400
    assert "strictly between" in res_end.json()["detail"]

    res_before = client.post(f"/api/projects/{project.id}/segments/{seg.id}/split?split_time=1.0")
    assert res_before.status_code == 400

    res_after = client.post(f"/api/projects/{project.id}/segments/{seg.id}/split?split_time=6.0")
    assert res_after.status_code == 400


@pytest.mark.asyncio
async def test_split_nonexistent_segment_and_project(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_nonexistent_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Nonexistent Proj", video_path=str(dummy_video))

    # DB call for non-existent segment -> returns empty list
    res_db = await db.split_segment(project.id, "nonexistent-seg-id", split_time=2.0)
    assert res_db == []

    # API call for non-existent segment -> returns HTTP 404
    client = TestClient(app)
    res_api_seg = client.post(f"/api/projects/{project.id}/segments/nonexistent-seg-id/split?split_time=2.0")
    assert res_api_seg.status_code == 404
    assert "Không tìm thấy" in res_api_seg.json()["detail"]

    # API call for non-existent project -> returns HTTP 404
    res_api_proj = client.post("/api/projects/unknown-proj-id/segments/some-seg/split?split_time=2.0")
    assert res_api_proj.status_code == 404


# ============================================================================
# 2. Word Timestamps Partitioning and Precision
# ============================================================================

@pytest.mark.asyncio
async def test_split_word_timestamps_partitioning(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_words_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Word Partition Proj", video_path=str(dummy_video))

    words = [
        WordTimestamp(word="one", start=0.5, end=1.5, probability=0.95),
        WordTimestamp(word="two", start=1.8, end=2.5, probability=0.92),
        WordTimestamp(word="three_right", start=2.8, end=3.4, probability=0.88),
        WordTimestamp(word="four_left", start=2.6, end=3.2, probability=0.85),
        WordTimestamp(word="five", start=3.5, end=4.5, probability=0.99),
    ]

    seg = await db.insert_segment(
        SubtitleSegment(
            project_id=project.id,
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="one two three_right four_left five",
            words=words,
        )
    )

    parts = await db.split_segment(
        project.id,
        seg.id,
        split_time=3.0,
        first_text="one two four_left",
        second_text="three_right five",
    )
    assert len(parts) == 2
    p1, p2 = parts[0], parts[1]

    # Verify segment 1 boundaries & words
    assert p1.start_time == 0.0
    assert p1.end_time == 3.0
    p1_words = p1.words
    assert len(p1_words) == 3
    assert [w.word for w in p1_words] == ["one", "two", "four_left"]
    # Check that four_left was clamped to split_t (3.0)
    assert p1_words[2].end == 3.0
    assert p1_words[2].start == 2.6
    for w in p1_words:
        assert w.start >= p1.start_time
        assert w.end <= p1.end_time
        assert w.end > w.start

    # Verify segment 2 boundaries & words
    assert p2.start_time == 3.0
    assert p2.end_time == 5.0
    p2_words = p2.words
    assert len(p2_words) == 2
    assert [w.word for w in p2_words] == ["three_right", "five"]
    # Check that three_right was clamped to split_t (3.0)
    assert p2_words[0].start == 3.0
    assert p2_words[0].end == 3.4
    for w in p2_words:
        assert w.start >= p2.start_time
        assert w.end <= p2.end_time
        assert w.end > w.start


@pytest.mark.asyncio
async def test_split_empty_words_and_single_word(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_empty_words_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Empty Words Proj", video_path=str(dummy_video))

    # Segment with no words and single word text
    seg = await db.insert_segment(
        SubtitleSegment(
            project_id=project.id,
            index=1,
            start_time=1.0,
            end_time=3.0,
            text="Hello",
            words=[],
        )
    )

    parts = await db.split_segment(project.id, seg.id, split_time=2.0)
    assert len(parts) == 2
    assert parts[0].words == []
    assert parts[1].words == []
    assert parts[0].text == "Hello"
    assert parts[1].text == ""
    assert parts[0].end_time == 2.0
    assert parts[1].start_time == 2.0


# ============================================================================
# 3. Cascading Splits: Contiguous Segment Indexes Verification
# ============================================================================

@pytest.mark.asyncio
async def test_cascading_splits_contiguous_indexes_no_holes_or_duplicates(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_cascade_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Cascade Proj", video_path=str(dummy_video))

    # Create 5 initial segments
    for i in range(1, 6):
        await db.insert_segment(
            SubtitleSegment(
                project_id=project.id,
                index=i,
                start_time=float(i * 10),
                end_time=float(i * 10 + 8),
                text=f"Segment {i}",
            )
        )

    # Verify initial indices 1..5
    proj = await db.get_project(project.id)
    assert [s.index for s in proj.segments] == [1, 2, 3, 4, 5]

    # Split segment 1 (10.0 - 18.0) at 14.0 -> total 6 segments
    s1_id = proj.segments[0].id
    await db.split_segment(project.id, s1_id, split_time=14.0)

    proj = await db.get_project(project.id)
    assert len(proj.segments) == 6
    assert [s.index for s in proj.segments] == [1, 2, 3, 4, 5, 6]

    # Split the last segment (now index 6, formerly 5: 50.0 - 58.0) at 54.0 -> total 7 segments
    s_last_id = proj.segments[-1].id
    await db.split_segment(project.id, s_last_id, split_time=54.0)

    proj = await db.get_project(project.id)
    assert len(proj.segments) == 7
    assert [s.index for s in proj.segments] == [1, 2, 3, 4, 5, 6, 7]

    # Split middle segment (index 3) at 24.0 -> total 8 segments
    s_mid_id = proj.segments[2].id
    await db.split_segment(project.id, s_mid_id, split_time=24.0)

    proj = await db.get_project(project.id)
    assert len(proj.segments) == 8
    assert [s.index for s in proj.segments] == [1, 2, 3, 4, 5, 6, 7, 8]

    # Inspect the raw SQLite database table directly to ensure no gaps or duplicate segment_index
    conn = await db.connect()
    try:
        cursor = await conn.execute(
            "SELECT segment_index FROM subtitle_segments WHERE project_id = ? ORDER BY segment_index ASC",
            (project.id,),
        )
        db_indices = [row[0] for row in await cursor.fetchall()]
        assert db_indices == [1, 2, 3, 4, 5, 6, 7, 8]
    finally:
        await conn.close()


# ============================================================================
# 4. Split and Merge Cycle Integrity
# ============================================================================

@pytest.mark.asyncio
async def test_split_and_merge_roundtrip(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_roundtrip_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Roundtrip Proj", video_path=str(dummy_video))

    words = [
        WordTimestamp(word="Xin", start=1.0, end=1.5),
        WordTimestamp(word="chào", start=1.6, end=2.0),
        WordTimestamp(word="Việt", start=2.5, end=3.0),
        WordTimestamp(word="Nam", start=3.1, end=3.8),
    ]

    orig = await db.insert_segment(
        SubtitleSegment(
            project_id=project.id,
            index=1,
            start_time=1.0,
            end_time=4.0,
            text="Xin chào Việt Nam",
            translated_text="Hello Vietnam",
            words=words,
        )
    )

    # Split at 2.2
    split_parts = await db.split_segment(
        project.id,
        orig.id,
        split_time=2.2,
        first_text="Xin chào",
        second_text="Việt Nam",
    )
    assert len(split_parts) == 2
    p1, p2 = split_parts[0], split_parts[1]

    # Merge them back
    merged = await db.merge_segments(project.id, p1.id, p2.id)
    assert merged is not None
    assert merged.start_time == 1.0
    assert merged.end_time == 4.0
    assert merged.text == "Xin chào Việt Nam"
    assert len(merged.words) == 4
    assert [w.word for w in merged.words] == ["Xin", "chào", "Việt", "Nam"]


# ============================================================================
# 5. Merge Boundary & Error Cases
# ============================================================================

@pytest.mark.asyncio
async def test_merge_boundary_and_error_cases(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "challenger_merge_error_test.sqlite3"
    monkeypatch.setattr(db, "db_path", test_db_path)
    await db.init_db()

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    project = await db.create_project(name="Merge Error Proj", video_path=str(dummy_video))
    s1 = await db.insert_segment(
        SubtitleSegment(project_id=project.id, index=1, start_time=1.0, end_time=2.0, text="One")
    )

    # 1. Merge same segment with itself -> should return None (fail safely)
    res_same = await db.merge_segments(project.id, s1.id, s1.id)
    assert res_same is None

    # 2. Merge non-existent segment -> should return None
    res_none = await db.merge_segments(project.id, s1.id, "fake-id-123")
    assert res_none is None

    # 3. Via API -> returns HTTP 400
    client = TestClient(app)
    api_same = client.post(f"/api/projects/{project.id}/segments/merge?first_id={s1.id}&second_id={s1.id}")
    assert api_same.status_code == 400
