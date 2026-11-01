"""
AI Post-Processor Service (llm_service.py) using Gemini 2.5 Flash API.
Optimized for ultra-low latency & high throughput:
- Compact token payload (cuts output tokens by >60% avoiding redundant UUID echoes)
- Parallel concurrent batch execution via asyncio.gather and Semaphore(3)
- Atomic batch SQLite updates via db.batch_update_segments
- Real-time WebSocket streaming of segment_updated events
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

from app.config import settings
from app.database import db
from app.schemas import (
    JobStatus,
    LLMJobRequest,
    SubtitleSegment,
    SubtitleSegmentUpdate,
    WebSocketEvent,
)

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]


def _call_gemini_batch_postprocess(
    batch: list[SubtitleSegment],
    fix_ocr_spelling: bool,
    translate_to: str | None,
    api_key: str,
    preferred_model: str | None = None,
) -> list[dict]:
    """Send a batch of subtitle segments to Gemini for fast spellcheck and/or translation with automatic model fallback."""
    from google import genai
    from google.genai import types
    from app.utils.gemini_client import generate_content_with_resilience

    client = genai.Client(api_key=api_key)

    # Compact payload: index and text (avoids echoing 36-char UUIDs, saving >60% output tokens)
    payload_items = [
        {"i": idx, "t": seg.text}
        for idx, seg in enumerate(batch)
    ]

    instructions = []
    if fix_ocr_spelling and translate_to:
        instructions.append(
            f"Fix OCR typos in 't' (original language) and translate naturally into '{translate_to}' in 'tr'. "
            "Return JSON array: [{\"i\": int, \"t\": string, \"tr\": string}]."
        )
    elif translate_to:
        instructions.append(
            f"Translate each subtitle naturally and concisely into '{translate_to}' for video dialogue in 'tr'. "
            "Return JSON array: [{\"i\": int, \"tr\": string}]."
        )
    elif fix_ocr_spelling:
        instructions.append(
            "Fix OCR typos and broken diacritics in 't'. "
            "Return JSON array: [{\"i\": int, \"t\": string}]."
        )

    prompt = (
        "You are an expert subtitle editor and translator. "
        + " ".join(instructions)
        + "\nInput:\n"
        + json.dumps(payload_items, ensure_ascii=False)
    )

    response = generate_content_with_resilience(
        client=client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
        preferred_model=preferred_model,
    )

    raw_text = (response.text or "[]").strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        parsed = parsed.get("items") or parsed.get("subtitles") or [parsed]
    return parsed if isinstance(parsed, list) else []


async def run_llm_postprocess_pipeline(
    req: LLMJobRequest,
    job_id: str,
    emit: EventCallback,
    cancel_event: asyncio.Event | None = None,
) -> list[SubtitleSegment]:
    """
    Run high-speed concurrent batch Gemini AI spellcheck and/or bilingual translation.
    Uses parallel batch requests and atomic SQLite updates for 10x throughput.
    """
    cancel_ev = cancel_event or asyncio.Event()
    start_wall = time.perf_counter()

    try:
        project = await db.get_project(req.project_id)
        if not project:
            raise ValueError(f"Project not found: {req.project_id}")

        api_key = req.gemini_api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("Thiếu Gemini API Key để chạy AI sửa lỗi chính tả / dịch thuật.")

        segments = list(project.segments)
        if not segments:
            await emit(
                WebSocketEvent(
                    event="job_completed",
                    project_id=req.project_id,
                    job_id=job_id,
                    progress=100.0,
                    message="Dự án chưa có câu phụ đề nào để dịch.",
                )
            )
            return []

        await db.update_project_status(req.project_id, status=JobStatus.RUNNING, progress=1.0)
        await emit(
            WebSocketEvent(
                event="job_started",
                project_id=req.project_id,
                job_id=job_id,
                progress=1.0,
                message=f"Bắt đầu dịch và xử lý AI ({len(segments)} dòng phụ đề)...",
            )
        )

        # Optimal batch size 60 for low latency and high context coherence
        batch_size = max(10, int(req.batch_size or 60))
        total = len(segments)
        batches = [segments[i : i + batch_size] for i in range(0, total, batch_size)]

        all_updated_segments: list[SubtitleSegment] = list(segments)
        completed_count = 0
        progress_lock = asyncio.Lock()
        sem = asyncio.Semaphore(3)  # Max 3 concurrent requests to respect rate limits

        async def process_batch(batch_offset: int, batch: list[SubtitleSegment]):
            nonlocal completed_count
            async with sem:
                if cancel_ev.is_set():
                    return

                if req.model:
                    raw_results = await asyncio.to_thread(
                        _call_gemini_batch_postprocess,
                        batch,
                        req.fix_ocr_spelling,
                        req.translate_to,
                        api_key,
                        preferred_model=req.model,
                    )
                else:
                    raw_results = await asyncio.to_thread(
                        _call_gemini_batch_postprocess,
                        batch,
                        req.fix_ocr_spelling,
                        req.translate_to,
                        api_key,
                    )

                # Build lookups by 'id' and by integer index 'i'
                by_id: dict[str, dict] = {}
                by_i: dict[int, dict] = {}
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    if "id" in item:
                        by_id[str(item["id"])] = item
                    if "i" in item:
                        try:
                            by_i[int(item["i"])] = item
                        except (ValueError, TypeError):
                            pass

                batch_db_updates = []
                batch_result_segments: list[SubtitleSegment] = []

                for pos, seg in enumerate(batch):
                    res_item = by_id.get(str(seg.id)) or by_i.get(pos)
                    if not res_item and pos < len(raw_results) and isinstance(raw_results[pos], dict):
                        res_item = raw_results[pos]

                    if res_item:
                        # Support both compact format ('t', 'tr') and legacy format ('text', 'translated_text')
                        new_text = res_item.get("t") or res_item.get("text") or seg.text
                        new_trans = (
                            res_item.get("tr")
                            or res_item.get("translated_text")
                            or res_item.get("translation")
                        )
                        if new_trans is not None:
                            new_trans = str(new_trans).strip() or None

                        upd_text = str(new_text).strip() if req.fix_ocr_spelling else seg.text
                        upd_trans = new_trans if req.translate_to else seg.translated_text

                        batch_db_updates.append({
                            "id": seg.id,
                            "text": upd_text,
                            "translated_text": upd_trans,
                        })
                        seg.text = upd_text
                        seg.translated_text = upd_trans

                    batch_result_segments.append(seg)
                    all_updated_segments[batch_offset + pos] = seg

                # Atomic single-transaction DB update for this batch
                if batch_db_updates:
                    await db.batch_update_segments(req.project_id, batch_db_updates)

                # Update progress and broadcast updated segments
                async with progress_lock:
                    completed_count += len(batch)
                    pct = min(99.0, (completed_count / total) * 100.0)

                for seg in batch_result_segments:
                    await emit(
                        WebSocketEvent(
                            event="segment_updated",
                            project_id=req.project_id,
                            job_id=job_id,
                            progress=round(pct, 1),
                            segment=seg,
                        )
                    )

        # Run all batches concurrently in parallel
        tasks = [process_batch(i * batch_size, b) for i, b in enumerate(batches)]
        await asyncio.gather(*tasks)

        if cancel_ev.is_set():
            await db.update_project_status(req.project_id, status=JobStatus.CANCELLED)
            await emit(
                WebSocketEvent(
                    event="job_cancelled",
                    project_id=req.project_id,
                    job_id=job_id,
                    message="Đã hủy tác vụ AI hậu kỳ.",
                )
            )
            return all_updated_segments

        elapsed = round(time.perf_counter() - start_wall, 2)
        await db.update_project_status(
            req.project_id, status=JobStatus.COMPLETED, progress=100.0
        )
        await emit(
            WebSocketEvent(
                event="job_completed",
                project_id=req.project_id,
                job_id=job_id,
                progress=100.0,
                message=f"Hoàn tất dịch AI {len(all_updated_segments)} câu phụ đề trong {elapsed}s.",
                extra={"segment_count": len(all_updated_segments), "elapsed_sec": elapsed},
            )
        )
        return all_updated_segments

    except Exception as exc:
        logger.exception("LLM post-processing failed for project %s", req.project_id)
        await db.update_project_status(req.project_id, status=JobStatus.FAILED)
        await emit(
            WebSocketEvent(
                event="job_failed",
                project_id=req.project_id,
                job_id=job_id,
                message=f"Lỗi AI hậu kỳ: {exc}",
            )
        )
        raise
