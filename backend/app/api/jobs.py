"""Job orchestration and WebSocket real-time streaming endpoints."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.database import db
from app.schemas import (
    JobStatus,
    LLMJobRequest,
    OCRJobRequest,
    STTJobRequest,
    WebSocketEvent,
)
from app.services.llm_service import run_llm_postprocess_pipeline
from app.services.ocr_pipeline import run_ocr_pipeline
from app.services.stt_pipeline import run_stt_pipeline


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobStreamManager:
    """Manages active WebSocket subscribers and running background tasks per project."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def connect(self, project_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[project_id].add(websocket)

    def disconnect(self, project_id: str, websocket: WebSocket) -> None:
        self._connections[project_id].discard(websocket)
        if not self._connections[project_id]:
            self._connections.pop(project_id, None)

    async def broadcast(self, event: WebSocketEvent) -> None:
        targets = list(self._connections.get(event.project_id, set()))
        if not targets:
            return

        payload = event.model_dump(mode="json")

        async def _safe_send(ws: WebSocket) -> WebSocket | None:
            try:
                await ws.send_json(payload)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(*[_safe_send(ws) for ws in targets], return_exceptions=True)
        for res in results:
            if isinstance(res, WebSocket):
                self.disconnect(event.project_id, res)

    def is_running(self, project_id: str) -> bool:
        task = self._tasks.get(project_id)
        return task is not None and not task.done()

    def register_job(self, project_id: str, task: asyncio.Task, cancel_ev: asyncio.Event) -> None:
        self._tasks[project_id] = task
        self._cancel_events[project_id] = cancel_ev

    def cancel_job(self, project_id: str) -> bool:
        cancelled = False
        ev = self._cancel_events.pop(project_id, None)
        if ev is not None:
            ev.set()
            cancelled = True
        task = self._tasks.pop(project_id, None)
        if task is not None and not task.done():
            task.cancel()
            cancelled = True
        return cancelled


stream_manager = JobStreamManager()


@router.websocket("/ws/{project_id}")
async def project_job_websocket(websocket: WebSocket, project_id: str) -> None:
    """Real-time WebSocket stream for job progress and subtitle segments."""
    await stream_manager.connect(project_id, websocket)
    try:
        # Send current job status immediately upon connection so frontend doesn't need to poll
        project = await db.get_project(project_id)
        if project:
            await websocket.send_json(
                {
                    "event": "job_status",
                    "project_id": project_id,
                    "status": project.status,
                    "progress": project.progress,
                    "segment_count": len(project.segments or []),
                }
            )

        while True:
            msg = await websocket.receive_text()
            if msg.strip().lower() == "ping":
                await websocket.send_json({"event": "pong", "project_id": project_id})
    except WebSocketDisconnect:
        stream_manager.disconnect(project_id, websocket)
    except Exception:
        stream_manager.disconnect(project_id, websocket)


@router.post("/stt")
async def start_stt_job(req: STTJobRequest) -> dict:
    """Launch an asynchronous Speech-to-Text extraction job for the specified project."""
    project = await db.get_project(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    if stream_manager.is_running(req.project_id):
        raise HTTPException(
            status_code=409,
            detail="Dự án này đang có một tiến trình trích xuất đang chạy.",
        )

    job_id = str(uuid.uuid4())
    cancel_ev = asyncio.Event()

    async def _runner() -> None:
        try:
            await run_stt_pipeline(
                req=req,
                job_id=job_id,
                emit=stream_manager.broadcast,
                cancel_event=cancel_ev,
            )
        except Exception as exc:
            logger.error("Background STT job %s failed: %s", job_id, exc)

    task = asyncio.create_task(_runner())
    stream_manager.register_job(req.project_id, task, cancel_ev)

    return {
        "job_id": job_id,
        "project_id": req.project_id,
        "status": JobStatus.RUNNING,
        "message": "Đã khởi chạy tiến trình Speech-to-Text.",
    }


@router.post("/ocr")
async def start_ocr_job(req: OCRJobRequest) -> dict:
    """Launch an asynchronous Hardsub OCR extraction job for the specified project."""
    project = await db.get_project(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    if stream_manager.is_running(req.project_id):
        raise HTTPException(
            status_code=409,
            detail="Dự án này đang có một tiến trình trích xuất đang chạy.",
        )

    job_id = str(uuid.uuid4())
    cancel_ev = asyncio.Event()

    async def _runner() -> None:
        try:
            await run_ocr_pipeline(
                req=req,
                job_id=job_id,
                emit=stream_manager.broadcast,
                cancel_event=cancel_ev,
            )
        except Exception as exc:
            logger.error("Background OCR job %s failed: %s", job_id, exc)

    task = asyncio.create_task(_runner())
    stream_manager.register_job(req.project_id, task, cancel_ev)

    return {
        "job_id": job_id,
        "project_id": req.project_id,
        "status": JobStatus.RUNNING,
        "message": "Đã khởi chạy tiến trình Hardsub OCR.",
    }


@router.post("/llm")
async def start_llm_job(req: LLMJobRequest) -> dict:
    """Launch an asynchronous Gemini AI spellcheck / bilingual translation job."""
    project = await db.get_project(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    if stream_manager.is_running(req.project_id):
        raise HTTPException(
            status_code=409,
            detail="Dự án này đang có một tiến trình đang chạy.",
        )

    job_id = str(uuid.uuid4())
    cancel_ev = asyncio.Event()

    async def _runner() -> None:
        try:
            await run_llm_postprocess_pipeline(
                req=req,
                job_id=job_id,
                emit=stream_manager.broadcast,
                cancel_event=cancel_ev,
            )
        except Exception as exc:
            logger.error("Background LLM job %s failed: %s", job_id, exc)
            try:
                await db.update_project_status(req.project_id, status=JobStatus.FAILED)
                await stream_manager.broadcast(
                    WebSocketEvent(
                        event="job_failed",
                        project_id=req.project_id,
                        job_id=job_id,
                        message=f"Lỗi AI hậu kỳ: {exc}",
                    )
                )
            except Exception:
                pass

    task = asyncio.create_task(_runner())
    stream_manager.register_job(req.project_id, task, cancel_ev)

    return {
        "job_id": job_id,
        "project_id": req.project_id,
        "status": JobStatus.RUNNING,
        "message": "Đã khởi chạy tiến trình AI hậu kỳ (Sửa lỗi / Dịch song ngữ).",
    }


@router.post("/{project_id}/cancel")
async def cancel_project_job(project_id: str) -> dict:
    """Request cancellation of a running STT/OCR job."""
    cancelled = stream_manager.cancel_job(project_id)
    await db.update_project_status(project_id, status=JobStatus.CANCELLED)
    await stream_manager.broadcast(
        WebSocketEvent(
            event="job_cancelled",
            project_id=project_id,
            message="Đã hủy tiến trình theo yêu cầu người dùng.",
        )
    )
    return {"project_id": project_id, "cancelled": True, "was_running": cancelled}
