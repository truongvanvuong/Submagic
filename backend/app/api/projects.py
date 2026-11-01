"""Project and Subtitle Segment CRUD API endpoints."""
from __future__ import annotations

import asyncio
import mimetypes
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database import db
from app.schemas import (
    ExtractionMode,
    JobStatus,
    ProjectCreateRequest,
    ProjectDetail,
    ProjectSummary,
    ROIConfig,
    SubtitleSegment,
    SubtitleSegmentUpdate,
)
from app.services.ffmpeg_service import extract_frame_jpeg, probe_video, resolve_media_path

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSummary])
async def list_projects() -> list[ProjectSummary]:
    return await db.list_projects()


@router.post("", response_model=ProjectDetail)
async def create_project_from_path(req: ProjectCreateRequest) -> ProjectDetail:
    video_path = resolve_media_path(req.video_path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file video: {video_path}")

    try:
        meta = probe_video(video_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc metadata video: {exc}") from exc

    name = req.name or video_path.stem
    return await db.create_project(
        name=name,
        video_path=str(video_path),
        video_metadata=meta,
        mode=req.mode,
        language=req.language,
    )


def save_upload_file(upload_file: UploadFile, dest_dir: Path) -> Path:
    """Save uploaded video file safely in worker thread without locking or blocking event loop."""
    safe_name = Path(upload_file.filename or "video.mp4").name
    # Clean problematic characters that interfere with URLs or filesystem operations
    cleaned_name = safe_name.replace("#", "_").replace("?", "_").replace("&", "_")
    target_path = dest_dir / cleaned_name

    # If file exists, generate unique timestamped path to prevent collision or overwrite locks
    if target_path.exists():
        import time
        target_path = dest_dir / f"{target_path.stem}_{int(time.time())}{target_path.suffix}"

    try:
        upload_file.file.seek(0)
    except Exception:
        pass

    try:
        with target_path.open("wb") as out_f:
            shutil.copyfileobj(upload_file.file, out_f, length=1024 * 1024)
    except (PermissionError, OSError):
        import time
        target_path = dest_dir / f"{target_path.stem}_{int(time.time())}{target_path.suffix}"
        try:
            upload_file.file.seek(0)
        except Exception:
            pass
        with target_path.open("wb") as out_f:
            shutil.copyfileobj(upload_file.file, out_f, length=1024 * 1024)

    return target_path


@router.post("/upload", response_model=ProjectDetail)
async def create_project_from_upload(
    file: UploadFile = File(...),
    mode: ExtractionMode = Form(ExtractionMode.STT),
    language: str = Form("auto"),
) -> ProjectDetail:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File tải lên không hợp lệ.")

    dest_dir = settings.projects_dir / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        dest_path = await asyncio.to_thread(save_upload_file, file, dest_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể lưu file video tải lên: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass

    try:
        meta = await asyncio.to_thread(probe_video, dest_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc metadata video: {exc}") from exc

    return await db.create_project(
        name=dest_path.stem,
        video_path=str(dest_path),
        video_metadata=meta,
        mode=mode,
        language=language,
    )


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str) -> ProjectDetail:
    from app.api.jobs import stream_manager

    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    if project.status in (JobStatus.RUNNING, JobStatus.QUEUED) and not stream_manager.is_running(project_id):
        await db.update_project_status(project_id, status=JobStatus.IDLE)
        project.status = JobStatus.IDLE
    return project


@router.get("/{project_id}/video")
async def stream_project_video(project_id: str) -> FileResponse:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    video_path = resolve_media_path(project.video_path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file video: {video_path}")

    mime_type, _ = mimetypes.guess_type(str(video_path))
    if not mime_type or not mime_type.startswith("video/"):
        mime_type = "video/mp4"

    return FileResponse(
        path=str(video_path),
        media_type=mime_type,
        filename=video_path.name,
        content_disposition_type="inline",
    )


@router.put("/{project_id}/roi", response_model=ProjectDetail)
async def update_project_roi(project_id: str, roi_config: ROIConfig) -> ProjectDetail:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    await db.update_project_status(
        project_id, status=project.status, roi_config=roi_config
    )
    updated = await db.get_project(project_id)
    assert updated is not None
    return updated


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict:
    deleted = await db.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    return {"deleted": True, "project_id": project_id}


@router.get("/{project_id}/frame-preview")
async def get_frame_preview(
    project_id: str,
    timestamp: float = Query(0.0, ge=0.0),
    crop_roi: bool = Query(False),
) -> Response:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    roi = project.roi_config if crop_roi else None
    try:
        jpeg_bytes = extract_frame_jpeg(project.video_path, timestamp, roi_config=roi)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/segments", response_model=SubtitleSegment)
async def add_manual_segment(
    project_id: str, update: SubtitleSegmentUpdate
) -> SubtitleSegment:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    seg = SubtitleSegment(
        project_id=project_id,
        index=len(project.segments) + 1,
        start_time=update.start_time or 0.0,
        end_time=update.end_time or 2.0,
        text=update.text or "",
        translated_text=update.translated_text,
        confidence=1.0,
        source="manual",
    )
    return await db.insert_segment(seg)


@router.patch("/{project_id}/segments/{segment_id}", response_model=SubtitleSegment)
async def update_segment(
    project_id: str, segment_id: str, update: SubtitleSegmentUpdate
) -> SubtitleSegment:
    updated = await db.update_segment(project_id, segment_id, update)
    if not updated:
        raise HTTPException(status_code=404, detail="Không tìm thấy dòng phụ đề.")
    return updated


@router.delete("/{project_id}/segments/{segment_id}")
async def delete_segment(project_id: str, segment_id: str) -> dict:
    ok = await db.delete_segment(project_id, segment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy dòng phụ đề.")
    return {"deleted": True, "segment_id": segment_id}


@router.post("/{project_id}/segments/merge", response_model=SubtitleSegment)
async def merge_two_segments(
    project_id: str,
    first_id: str = Query(...),
    second_id: str = Query(...),
) -> SubtitleSegment:
    merged = await db.merge_segments(project_id, first_id, second_id)
    if not merged:
        raise HTTPException(status_code=400, detail="Không thể gộp 2 dòng phụ đề đã chọn.")
    return merged


class SplitSegmentRequest(BaseModel):
    split_time: float = Field(..., description="Thời điểm tách (giây)")
    first_text: str | None = None
    second_text: str | None = None


@router.post(
    "/{project_id}/segments/{segment_id}/split",
    response_model=list[SubtitleSegment],
)
async def split_segment(
    project_id: str,
    segment_id: str,
    req: SplitSegmentRequest | None = None,
    split_time: float | None = Query(None),
    first_text: str | None = Query(None),
    second_text: str | None = Query(None),
) -> list[SubtitleSegment]:
    st = req.split_time if (req is not None and req.split_time is not None) else split_time
    if st is None:
        raise HTTPException(status_code=400, detail="split_time là bắt buộc.")

    t1 = req.first_text if (req is not None and req.first_text is not None) else first_text
    t2 = req.second_text if (req is not None and req.second_text is not None) else second_text

    try:
        segments = await db.split_segment(
            project_id=project_id,
            segment_id=segment_id,
            split_time=st,
            first_text=t1,
            second_text=t2,
        )
        if not segments:
            raise HTTPException(status_code=404, detail="Không tìm thấy dòng phụ đề.")
        return segments
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

