"""API routes for managing extraction history."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.database import db

router = APIRouter(prefix="/api/history", tags=["history"])


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
async def download_history_file(history_id: str) -> FileResponse:
    """Download exported subtitle file corresponding to a history record."""
    records = await db.list_history()
    target = next((r for r in records if r["id"] == history_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử.")

    srt_path = target.get("export_srt_path")
    if not srt_path or not Path(srt_path).exists():
        raise HTTPException(status_code=404, detail="File phụ đề đã bị xóa hoặc không tồn tại.")

    p = Path(srt_path)
    safe_name = quote(p.name)
    return FileResponse(
        path=str(p),
        media_type="application/x-subrip; charset=utf-8",
        filename=p.name,
        headers={"Content-Disposition": f'attachment; filename="{p.name}"; filename*=UTF-8\'\'{safe_name}'},
    )
