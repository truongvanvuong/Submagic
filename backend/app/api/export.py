"""Export routes for downloading or saving .SRT, .VTT, and .TXT subtitle files."""
from __future__ import annotations

import asyncio
from pathlib import Path
import re
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Response

from app.config import settings
from app.database import db
from app.schemas import BilingualLayout, ExportFormat, ExportRequest
from app.utils.subtitle_formatter import format_subtitles

_orig_get_event_loop = asyncio.get_event_loop


def _safe_get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = _orig_get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


asyncio.get_event_loop = _safe_get_event_loop


def make_safe_header_path(path: Path | str) -> str:
    path_str = str(path)
    try:
        path_str.encode("latin-1")
        return path_str
    except UnicodeEncodeError:
        return quote(path_str.encode("utf-8"))


router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["export"])

MEDIA_TYPES: dict[ExportFormat, str] = {
    ExportFormat.SRT: "application/x-subrip; charset=utf-8",
    ExportFormat.VTT: "text/vtt; charset=utf-8",
    ExportFormat.TXT: "text/plain; charset=utf-8",
}


def make_content_disposition(safe_name: str) -> str:
    encoded = quote(safe_name)
    try:
        safe_name.encode("latin-1")
        if encoded == safe_name:
            return f'attachment; filename="{safe_name}"'
        return f'attachment; filename="{safe_name}"; filename*=UTF-8\'\'{encoded}'
    except UnicodeEncodeError:
        ascii_fallback = safe_name.encode("ascii", "replace").decode("ascii")
        return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


@router.get("")
async def export_project_subtitles_get(
    project_id: str,
    format: ExportFormat = Query(ExportFormat.SRT),
    bilingual_layout: BilingualLayout = Query(BilingualLayout.NONE),
    include_timestamps_in_txt: bool = Query(True),
) -> Response:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    content = format_subtitles(
        segments=project.segments,
        export_format=format,
        bilingual_layout=bilingual_layout,
        include_timestamps_in_txt=include_timestamps_in_txt,
    )

    ext = format.value
    sanitized_title = re.sub(r'[\\/*?:"<>|]', '_', project.name).strip() or "subtitles"
    safe_name = f"{sanitized_title}.{ext}"
    out_path = settings.exports_dir / safe_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    media_type = MEDIA_TYPES.get(format, "text/plain; charset=utf-8")
    content_disp = make_content_disposition(safe_name)

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={
            "Content-Disposition": content_disp,
            "X-Saved-Path": make_safe_header_path(out_path),
        },
    )


@router.post("")
async def export_project_subtitles(project_id: str, req: ExportRequest) -> Response:
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    content = format_subtitles(
        segments=project.segments,
        export_format=req.format,
        bilingual_layout=req.bilingual_layout,
        include_timestamps_in_txt=req.include_timestamps_in_txt,
    )

    ext = req.format.value
    sanitized_title = re.sub(r'[\\/*?:"<>|]', '_', project.name).strip() or "subtitles"
    safe_name = f"{sanitized_title}.{ext}"
    if req.save_to_disk_path:
        out_path = Path(req.save_to_disk_path).resolve()
    else:
        out_path = settings.exports_dir / safe_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    media_type = MEDIA_TYPES.get(req.format, "text/plain; charset=utf-8")
    content_disp = make_content_disposition(safe_name)

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={
            "Content-Disposition": content_disp,
            "X-Saved-Path": make_safe_header_path(out_path),
        },
    )
