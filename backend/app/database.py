"""SQLite persistence layer with WAL mode and real-time auto-save support."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from app.config import settings
from app.schemas import (
    ExtractionMode,
    JobStatus,
    ProjectDetail,
    ProjectSummary,
    ROIConfig,
    SubtitleSegment,
    SubtitleSegmentUpdate,
    VideoMetadata,
    WordTimestamp,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Async SQLite manager for projects, subtitle segments, and app settings."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path

    async def connect(self) -> aiosqlite.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self.db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    async def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = await self.connect()
        try:
            await conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    video_metadata_json TEXT,
                    mode TEXT NOT NULL DEFAULT 'stt',
                    language TEXT,
                    roi_config_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    progress REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS subtitle_segments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    text TEXT NOT NULL,
                    translated_text TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'stt',
                    crop_image_url TEXT,
                    words_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_segments_project_idx
                    ON subtitle_segments(project_id, segment_index);

                CREATE INDEX IF NOT EXISTS idx_segments_project_time
                    ON subtitle_segments(project_id, start_time);

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS extraction_history (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    language TEXT,
                    detected_language TEXT,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    export_srt_path TEXT,
                    export_vtt_path TEXT,
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    elapsed_sec REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_history_created
                    ON extraction_history(created_at DESC);
                """
            )
            await conn.commit()
        finally:
            await conn.close()

    async def create_project(
        self,
        name: str,
        video_path: str,
        video_metadata: VideoMetadata | None = None,
        mode: ExtractionMode = ExtractionMode.STT,
        language: str | None = "vi",
        roi_config: ROIConfig | None = None,
    ) -> ProjectDetail:
        project_id = str(uuid.uuid4())
        now = utc_now_iso()
        roi = roi_config or ROIConfig()
        meta_json = video_metadata.model_dump_json() if video_metadata else None
        roi_json = roi.model_dump_json()

        conn = await self.connect()
        try:
            await conn.execute(
                """
                INSERT INTO projects (
                    id, name, video_path, video_metadata_json, mode, language,
                    roi_config_json, status, progress, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    name,
                    video_path,
                    meta_json,
                    mode.value,
                    language,
                    roi_json,
                    JobStatus.IDLE.value,
                    0.0,
                    now,
                    now,
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

        return ProjectDetail(
            id=project_id,
            name=name,
            video_path=video_path,
            video_metadata=video_metadata,
            mode=mode,
            language=language,
            roi_config=roi,
            status=JobStatus.IDLE,
            progress=0.0,
            segment_count=0,
            created_at=now,
            updated_at=now,
            segments=[],
        )

    def _row_to_project_summary(self, row: aiosqlite.Row, segment_count: int = 0) -> ProjectSummary:
        meta = (
            VideoMetadata.model_validate_json(row["video_metadata_json"])
            if row["video_metadata_json"]
            else None
        )
        roi = (
            ROIConfig.model_validate_json(row["roi_config_json"])
            if row["roi_config_json"]
            else ROIConfig()
        )
        return ProjectSummary(
            id=row["id"],
            name=row["name"],
            video_path=row["video_path"],
            video_metadata=meta,
            mode=ExtractionMode(row["mode"]),
            language=row["language"],
            roi_config=roi,
            status=JobStatus(row["status"]),
            progress=float(row["progress"]),
            segment_count=segment_count,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_segment(self, row: aiosqlite.Row) -> SubtitleSegment:
        words_raw = json.loads(row["words_json"]) if row["words_json"] else []
        words = [WordTimestamp.model_validate(w) for w in words_raw]
        return SubtitleSegment(
            id=row["id"],
            project_id=row["project_id"],
            index=int(row["segment_index"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            text=row["text"],
            translated_text=row["translated_text"],
            confidence=float(row["confidence"]),
            source=row["source"],
            crop_image_url=row["crop_image_url"],
            words=words,
        )

    async def list_projects(self) -> list[ProjectSummary]:
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                """
                SELECT p.*, COUNT(s.id) AS seg_count
                FROM projects p
                LEFT JOIN subtitle_segments s ON s.project_id = p.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC
                """
            )
            rows = await cursor.fetchall()
            return [self._row_to_project_summary(row, int(row["seg_count"])) for row in rows]
        finally:
            await conn.close()

    async def get_project(self, project_id: str) -> ProjectDetail | None:
        conn = await self.connect()
        try:
            cursor = await conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = await cursor.fetchone()
            if not row:
                return None

            seg_cursor = await conn.execute(
                """
                SELECT * FROM subtitle_segments
                WHERE project_id = ?
                ORDER BY start_time ASC, segment_index ASC
                """,
                (project_id,),
            )
            seg_rows = await seg_cursor.fetchall()
            segments = [self._row_to_segment(r) for r in seg_rows]
            # Normalize sequential 1-based indices
            for idx, seg in enumerate(segments, start=1):
                seg.index = idx

            summary = self._row_to_project_summary(row, len(segments))
            return ProjectDetail(
                **summary.model_dump(),
                segments=segments,
            )
        finally:
            await conn.close()

    async def update_project_status(
        self,
        project_id: str,
        status: JobStatus,
        progress: float | None = None,
        mode: ExtractionMode | None = None,
        roi_config: ROIConfig | None = None,
    ) -> None:
        now = utc_now_iso()
        conn = await self.connect()
        try:
            fields = ["status = ?", "updated_at = ?"]
            params: list[Any] = [status.value, now]
            if progress is not None:
                fields.append("progress = ?")
                params.append(round(float(progress), 2))
            if mode is not None:
                fields.append("mode = ?")
                params.append(mode.value)
            if roi_config is not None:
                fields.append("roi_config_json = ?")
                params.append(roi_config.model_dump_json())
            params.append(project_id)

            sql = f"UPDATE projects SET {', '.join(fields)} WHERE id = ?"
            await conn.execute(sql, tuple(params))
            await conn.commit()
        finally:
            await conn.close()

    async def delete_project(self, project_id: str) -> bool:
        conn = await self.connect()
        try:
            cursor = await conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def clear_project_segments(self, project_id: str, source: str | None = None) -> None:
        conn = await self.connect()
        try:
            if source:
                await conn.execute(
                    "DELETE FROM subtitle_segments WHERE project_id = ? AND source = ?",
                    (project_id, source),
                )
            else:
                await conn.execute(
                    "DELETE FROM subtitle_segments WHERE project_id = ?",
                    (project_id,),
                )
            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), project_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def insert_segment(self, segment: SubtitleSegment) -> SubtitleSegment:
        """Insert a single subtitle segment immediately (real-time auto-save)."""
        seg_id = segment.id or str(uuid.uuid4())
        now = utc_now_iso()
        words_json = json.dumps([w.model_dump() for w in segment.words], ensure_ascii=False)

        conn = await self.connect()
        try:
            await conn.execute(
                """
                INSERT INTO subtitle_segments (
                    id, project_id, segment_index, start_time, end_time,
                    text, translated_text, confidence, source, crop_image_url,
                    words_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg_id,
                    segment.project_id,
                    segment.index,
                    round(float(segment.start_time), 3),
                    round(float(segment.end_time), 3),
                    segment.text,
                    segment.translated_text,
                    round(float(segment.confidence), 4),
                    segment.source,
                    segment.crop_image_url,
                    words_json,
                    now,
                ),
            )
            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, segment.project_id),
            )
            await conn.commit()
        finally:
            await conn.close()

        segment.id = seg_id
        return segment

    async def update_segment(
        self, project_id: str, segment_id: str, update: SubtitleSegmentUpdate
    ) -> SubtitleSegment | None:
        now = utc_now_iso()
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                "SELECT * FROM subtitle_segments WHERE id = ? AND project_id = ?",
                (segment_id, project_id),
            )
            row = await cursor.fetchone()
            if not row:
                return None

            current = self._row_to_segment(row)
            new_start = update.start_time if update.start_time is not None else current.start_time
            new_end = update.end_time if update.end_time is not None else current.end_time
            new_text = update.text if update.text is not None else current.text
            new_trans = (
                update.translated_text
                if update.translated_text is not None
                else current.translated_text
            )
            new_conf = (
                round(float(update.confidence), 4)
                if update.confidence is not None
                else current.confidence
            )
            new_crop = (
                update.crop_image_url
                if update.crop_image_url is not None
                else current.crop_image_url
            )

            await conn.execute(
                """
                UPDATE subtitle_segments
                SET start_time = ?, end_time = ?, text = ?, translated_text = ?,
                    confidence = ?, crop_image_url = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (
                    round(float(new_start), 3),
                    round(float(new_end), 3),
                    new_text,
                    new_trans,
                    new_conf,
                    new_crop,
                    now,
                    segment_id,
                    project_id,
                ),
            )
            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            await conn.commit()

            current.start_time = round(float(new_start), 3)
            current.end_time = round(float(new_end), 3)
            current.text = new_text
            current.translated_text = new_trans
            current.confidence = new_conf
            current.crop_image_url = new_crop
            return current

        finally:
            await conn.close()

    async def batch_update_segments(
        self,
        project_id: str,
        updates: list[dict],
    ) -> None:
        """Bulk update text and/or translated_text across multiple segments in a single transaction."""
        if not updates:
            return
        now = utc_now_iso()
        conn = await self.connect()
        try:
            await conn.execute("BEGIN TRANSACTION")
            for u in updates:
                seg_id = u["id"]
                new_text = u.get("text")
                new_trans = u.get("translated_text")
                if new_text is not None and new_trans is not None:
                    await conn.execute(
                        """
                        UPDATE subtitle_segments
                        SET text = ?, translated_text = ?, updated_at = ?
                        WHERE id = ? AND project_id = ?
                        """,
                        (new_text, new_trans, now, seg_id, project_id),
                    )
                elif new_trans is not None:
                    await conn.execute(
                        """
                        UPDATE subtitle_segments
                        SET translated_text = ?, updated_at = ?
                        WHERE id = ? AND project_id = ?
                        """,
                        (new_trans, now, seg_id, project_id),
                    )
                elif new_text is not None:
                    await conn.execute(
                        """
                        UPDATE subtitle_segments
                        SET text = ?, updated_at = ?
                        WHERE id = ? AND project_id = ?
                        """,
                        (new_text, now, seg_id, project_id),
                    )

            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()

    async def delete_segment(self, project_id: str, segment_id: str) -> bool:
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                "DELETE FROM subtitle_segments WHERE id = ? AND project_id = ?",
                (segment_id, project_id),
            )
            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), project_id),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def merge_segments(
        self, project_id: str, first_segment_id: str, second_segment_id: str
    ) -> SubtitleSegment | None:
        """Merge two segments into one combined segment."""
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                """
                SELECT * FROM subtitle_segments
                WHERE project_id = ? AND id IN (?, ?)
                ORDER BY start_time ASC
                """,
                (project_id, first_segment_id, second_segment_id),
            )
            rows = await cursor.fetchall()
            if len(rows) != 2:
                return None

            s1 = self._row_to_segment(rows[0])
            s2 = self._row_to_segment(rows[1])

            merged_text = f"{s1.text.strip()} {s2.text.strip()}".strip()
            merged_trans = None
            if s1.translated_text or s2.translated_text:
                merged_trans = (
                    f"{(s1.translated_text or '').strip()} {(s2.translated_text or '').strip()}".strip()
                )
            merged_words = s1.words + s2.words
            merged_conf = round((s1.confidence + s2.confidence) / 2.0, 4)
            now = utc_now_iso()

            await conn.execute(
                """
                UPDATE subtitle_segments
                SET start_time = ?, end_time = ?, text = ?, translated_text = ?,
                    confidence = ?, words_json = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (
                    min(s1.start_time, s2.start_time),
                    max(s1.end_time, s2.end_time),
                    merged_text,
                    merged_trans,
                    merged_conf,
                    json.dumps([w.model_dump() for w in merged_words], ensure_ascii=False),
                    now,
                    s1.id,
                    project_id,
                ),
            )
            await conn.execute(
                "DELETE FROM subtitle_segments WHERE id = ? AND project_id = ?",
                (s2.id, project_id),
            )
            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            await conn.commit()

            s1.start_time = min(s1.start_time, s2.start_time)
            s1.end_time = max(s1.end_time, s2.end_time)
            s1.text = merged_text
            s1.translated_text = merged_trans
            s1.confidence = merged_conf
            s1.words = merged_words
            return s1
        finally:
            await conn.close()

    async def split_segment(
        self,
        project_id: str,
        segment_id: str,
        split_time: float,
        first_text: str | None = None,
        second_text: str | None = None,
    ) -> list[SubtitleSegment]:
        """Split a subtitle segment at split_time into two consecutive segments."""
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                "SELECT * FROM subtitle_segments WHERE project_id = ? AND id = ?",
                (project_id, segment_id),
            )
            row = await cursor.fetchone()
            if not row:
                return []

            orig = self._row_to_segment(row)
            split_t = round(float(split_time), 3)

            if split_t <= orig.start_time or split_t >= orig.end_time:
                raise ValueError(
                    f"split_time ({split_t}) must be strictly between start_time ({orig.start_time}) and end_time ({orig.end_time})."
                )

            # Determine text for both parts
            if first_text is not None and second_text is not None:
                t1 = first_text.strip()
                t2 = second_text.strip()
            elif first_text is not None:
                t1 = first_text.strip()
                t2 = (orig.text or "").replace(t1, "", 1).strip()
            elif second_text is not None:
                t2 = second_text.strip()
                t1 = (orig.text or "").replace(t2, "", 1).strip()
            else:
                words = (orig.text or "").split()
                if len(words) >= 2:
                    duration = max(0.001, orig.end_time - orig.start_time)
                    ratio = (split_t - orig.start_time) / duration
                    split_idx = max(1, min(len(words) - 1, round(len(words) * ratio)))
                    t1 = " ".join(words[:split_idx]).strip()
                    t2 = " ".join(words[split_idx:]).strip()
                else:
                    t1 = orig.text or ""
                    t2 = ""

            # Partition word timestamps if present
            words1: list[WordTimestamp] = []
            words2: list[WordTimestamp] = []
            for w in orig.words:
                if w.end <= split_t:
                    words1.append(w)
                elif w.start >= split_t:
                    words2.append(w)
                else:
                    if (split_t - w.start) >= (w.end - split_t):
                        words1.append(
                            WordTimestamp(
                                word=w.word,
                                start=w.start,
                                end=split_t,
                                probability=w.probability,
                            )
                        )
                    else:
                        words2.append(
                            WordTimestamp(
                                word=w.word,
                                start=split_t,
                                end=w.end,
                                probability=w.probability,
                            )
                        )

            now = utc_now_iso()
            new_id = str(uuid.uuid4())

            # Shift segment_index of subsequent segments
            await conn.execute(
                """
                UPDATE subtitle_segments
                SET segment_index = segment_index + 1
                WHERE project_id = ? AND segment_index > ?
                """,
                (project_id, orig.index),
            )

            # Update the original segment (first half)
            await conn.execute(
                """
                UPDATE subtitle_segments
                SET end_time = ?, text = ?, translated_text = ?,
                    words_json = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (
                    split_t,
                    t1,
                    orig.translated_text,
                    json.dumps([w.model_dump() for w in words1], ensure_ascii=False),
                    now,
                    orig.id,
                    project_id,
                ),
            )

            # Insert the new segment (second half)
            await conn.execute(
                """
                INSERT INTO subtitle_segments (
                    id, project_id, segment_index, start_time, end_time,
                    text, translated_text, confidence, source, crop_image_url,
                    words_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    project_id,
                    orig.index + 1,
                    split_t,
                    orig.end_time,
                    t2,
                    None,
                    orig.confidence,
                    orig.source,
                    orig.crop_image_url,
                    json.dumps([w.model_dump() for w in words2], ensure_ascii=False),
                    now,
                ),
            )

            await conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            await conn.commit()

            first_seg = SubtitleSegment(
                id=orig.id,
                project_id=project_id,
                index=orig.index,
                start_time=orig.start_time,
                end_time=split_t,
                text=t1,
                translated_text=orig.translated_text,
                confidence=orig.confidence,
                source=orig.source,
                crop_image_url=orig.crop_image_url,
                words=words1,
            )

            second_seg = SubtitleSegment(
                id=new_id,
                project_id=project_id,
                index=orig.index + 1,
                start_time=split_t,
                end_time=orig.end_time,
                text=t2,
                translated_text=None,
                confidence=orig.confidence,
                source=orig.source,
                crop_image_url=orig.crop_image_url,
                words=words2,
            )

            return [first_seg, second_seg]
        finally:
            await conn.close()

    async def add_history_record(
        self,
        project_id: str,
        mode: str,
        language: str | None = None,
        detected_language: str | None = None,
        segment_count: int = 0,
        export_srt_path: str | None = None,
        export_vtt_path: str | None = None,
        duration_sec: float = 0.0,
        elapsed_sec: float = 0.0,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Insert a permanent extraction history record."""
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                "SELECT name, video_path FROM projects WHERE id = ?", (project_id,)
            )
            row = await cursor.fetchone()
            project_name = row["name"] if row else "Unknown Video"
            video_path = row["video_path"] if row else ""

            hist_id = str(uuid.uuid4())
            now = utc_now_iso()
            await conn.execute(
                """
                INSERT INTO extraction_history (
                    id, project_id, project_name, video_path, mode, language,
                    detected_language, segment_count, export_srt_path, export_vtt_path,
                    duration_sec, elapsed_sec, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hist_id,
                    project_id,
                    project_name,
                    video_path,
                    mode,
                    language,
                    detected_language,
                    segment_count,
                    export_srt_path,
                    export_vtt_path,
                    round(duration_sec, 2),
                    round(elapsed_sec, 2),
                    status,
                    now,
                ),
            )
            await conn.commit()
            return {
                "id": hist_id,
                "project_id": project_id,
                "project_name": project_name,
                "video_path": video_path,
                "mode": mode,
                "language": language,
                "detected_language": detected_language,
                "segment_count": segment_count,
                "export_srt_path": export_srt_path,
                "export_vtt_path": export_vtt_path,
                "duration_sec": duration_sec,
                "elapsed_sec": elapsed_sec,
                "status": status,
                "created_at": now,
            }
        finally:
            await conn.close()

    async def list_history(self) -> list[dict[str, Any]]:
        """List extraction history ordered newest first."""
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                """
                SELECT h.*, p.name as live_project_name
                FROM extraction_history h
                LEFT JOIN projects p ON p.id = h.project_id
                ORDER BY h.created_at DESC
                """
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "project_id": r["project_id"],
                    "project_name": r["live_project_name"] or r["project_name"],
                    "video_path": r["video_path"],
                    "mode": r["mode"],
                    "language": r["language"],
                    "detected_language": r["detected_language"],
                    "segment_count": r["segment_count"],
                    "export_srt_path": r["export_srt_path"],
                    "export_vtt_path": r["export_vtt_path"],
                    "duration_sec": float(r["duration_sec"]),
                    "elapsed_sec": float(r["elapsed_sec"]),
                    "status": r["status"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            await conn.close()

    async def delete_history_record(self, history_id: str) -> bool:
        """Delete an extraction history item."""
        conn = await self.connect()
        try:
            cursor = await conn.execute(
                "DELETE FROM extraction_history WHERE id = ?", (history_id,)
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()


db = DatabaseManager()
