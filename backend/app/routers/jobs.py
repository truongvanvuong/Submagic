"""Job orchestration and WebSocket real-time streaming endpoints."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.database import db
from app.schemas import (
    GeminiOCREstimateRequest,
    GeminiOCREstimateResponse,
    GeminiOCRExecuteRequest,
    JobStatus,
    LLMEstimateRequest,
    LLMEstimateResponse,
    LLMJobRequest,
    OCRJobRequest,
    STTJobRequest,
    SubtitleSegmentUpdate,
    WebSocketEvent,
)
from app.services.llm_service import estimate_llm_cost, run_llm_postprocess_pipeline
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


@router.post("/estimate-llm", response_model=LLMEstimateResponse)
async def estimate_llm_job(req: LLMEstimateRequest) -> LLMEstimateResponse:
    """Estimate token usage and cost for an LLM translation/polish job before execution."""
    try:
        return await estimate_llm_cost(req)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


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


@router.delete("/translation-cache")
@router.post("/translation-cache/clear")
async def clear_translation_cache_endpoint(
    target_lang: str | None = Query(None),
    context_type: str | None = Query(None),
    model_name: str | None = Query(None),
) -> dict:
    """Clear persistent translation cache entries with optional filters."""
    deleted_rows = await db.clear_translation_cache(
        target_lang=target_lang,
        context_type=context_type,
        model_name=model_name,
    )
    return {
        "cleared": True,
        "deleted_rows": deleted_rows,
        "target_lang": target_lang,
        "context_type": context_type,
        "model_name": model_name,
    }


@router.post("/estimate-gemini-ocr", response_model=GeminiOCREstimateResponse)
async def estimate_gemini_ocr_endpoint(req: GeminiOCREstimateRequest) -> GeminiOCREstimateResponse:
    """Estimate token usage and cost for running Gemini Vision OCR exclusively on low-confidence segments."""
    project = await db.get_project(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    total_segments = len(project.segments)
    low_conf_segs = [s for s in project.segments if s.is_low_confidence or s.confidence < 0.60]
    num_low_conf = len(low_conf_segs)
    lines_to_process = min(num_low_conf, req.max_lines)

    # Standard Gemini Flash Vision token calculation for cropped subtitle regions:
    # ~280 prompt tokens per image crop + context prompt, ~30 completion tokens
    prompt_tokens = lines_to_process * 280
    completion_tokens = lines_to_process * 30
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = (prompt_tokens * 0.000000075) + (completion_tokens * 0.00000030)

    return GeminiOCREstimateResponse(
        total_segments=total_segments,
        low_confidence_segments=num_low_conf,
        lines_to_process=lines_to_process,
        estimated_prompt_tokens=prompt_tokens,
        estimated_completion_tokens=completion_tokens,
        estimated_total_tokens=total_tokens,
        estimated_cost_usd=round(cost_usd, 6),
    )


@router.post("/gemini-ocr-fallback")
async def execute_gemini_ocr_fallback(req: GeminiOCRExecuteRequest) -> dict:
    """Execute Gemini Vision OCR fallback exclusively on segments flagged as low confidence."""
    project = await db.get_project(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    low_conf_segs = [s for s in project.segments if s.is_low_confidence or s.confidence < 0.60]
    targets = low_conf_segs[: req.max_lines]

    from app.services.ocr_engines import ocr_engine_cache
    from app.utils.cjk_normalizer import convert_t2s
    import cv2

    processed = 0
    updated_segments = []

    for seg in targets:
        # Load cropped image if available
        crop_path = None
        if seg.crop_image_url:
            raw_url = seg.crop_image_url.replace("/media/", "")
            crop_path = settings.crops_dir / raw_url

        if crop_path and crop_path.exists():
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is not None:
                try:
                    gem_res = ocr_engine_cache.recognize_gemini(
                        img_bgr,
                        api_key=req.gemini_api_key,
                        preferred_model=req.model,
                    )
                    if gem_res.text:
                        new_text = gem_res.text.strip()
                        norm_text = convert_t2s(new_text)
                        await db.update_segment(
                            req.project_id,
                            seg.id,
                            SubtitleSegmentUpdate(
                                text=new_text,
                                original_text=new_text,
                                normalized_text=norm_text,
                                confidence=max(seg.confidence, gem_res.confidence or 0.95),
                                is_low_confidence=False,
                            ),
                        )
                        seg.text = new_text
                        seg.is_low_confidence = False
                        processed += 1
                        updated_segments.append(seg.id)
                except Exception as exc:
                    logger.warning("Gemini OCR fallback failed for segment %s: %s", seg.id, exc)

    return {
        "project_id": req.project_id,
        "processed_segments": processed,
        "updated_segment_ids": updated_segments,
    }
