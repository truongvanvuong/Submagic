"""API routes for managing extraction history."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response

from app.config import settings
from app.database import db

router = APIRouter(prefix="/api/history", tags=["history"])


def make_content_disposition(safe_name: str) -> str:
    """Format RFC 5987 / RFC 6266 Content-Disposition header safely for any Unicode filename."""
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
async def get_history() -> list[dict[str, Any]]:
    """Return all past extraction history records sorted newest first."""
    return await db.list_history()


@router.delete("/{history_id}")
async def delete_history(history_id: str) -> dict[str, Any]:
    """Delete a specific history record."""
    ok = await db.delete_history_record(history_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử.")
    return {"deleted": True, "history_id": history_id}


@router.get("/{history_id}/download")
async def download_history_file(history_id: str) -> Response:
    """Download exported subtitle file corresponding to a history record safely."""
    if not history_id or not history_id.strip():
        raise HTTPException(status_code=400, detail="Mã bản ghi lịch sử không hợp lệ.")

    records = await db.list_history()
    target = next((r for r in records if r["id"] == history_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử.")

    srt_path = target.get("export_srt_path")
    if not srt_path:
        raise HTTPException(status_code=404, detail="Bản ghi lịch sử này không có file phụ đề để tải.")

    p = Path(srt_path)
    if not p.exists():
        fallback_p = settings.exports_dir / p.name
        if fallback_p.exists():
            p = fallback_p
        else:
            raise HTTPException(status_code=404, detail="File phụ đề đã bị xóa hoặc không tồn tại trên hệ thống.")

    try:
        content_bytes = p.read_bytes()
    except Exception as read_err:
        raise HTTPException(status_code=500, detail=f"Không thể đọc file phụ đề: {read_err}")

    content_disp = make_content_disposition(p.name)
    return Response(
        content=content_bytes,
        media_type="application/x-subrip; charset=utf-8",
        headers={
            "Content-Disposition": content_disp,
        },
    )
