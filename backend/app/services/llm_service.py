"""
AI Post-Processor & Translation Service (llm_service.py) using Google Gemini API.
Features:
- Multi-context translation: Cổ trang (historical), Hiện đại (modern), Đời thường (casual).
- Context window batching: Includes adjacent preceding/succeeding lines for dialogue consistency.
- Granular retry: Verifies row count match and only retries missing/failed lines instead of whole batch.
- Subtitle duration constraint: Enforces concise translations matching on-screen display duration.
- Trailing period removal: Automatically strips unnecessary trailing '.' via remove_trailing_period.
- Cost control & Caching: Persistent translation cache (SQLite), incremental translation (skips existing lines).
- Token usage tracking & Estimation: Logs token usage per request and provides pre-translation estimation.
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
    LLMEstimateRequest,
    LLMEstimateResponse,
    LLMJobRequest,
    SubtitleSegment,
    SubtitleSegmentUpdate,
    TranslationContext,
    WebSocketEvent,
)
from app.utils.subtitle_formatter import remove_trailing_period

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]


def _get_context_instruction(context: TranslationContext | str | None) -> str:
    """Return specific tone, pronoun, and vocabulary rules for the chosen translation context."""
    if hasattr(context, "value"):
        ctx = str(context.value).lower()
    else:
        ctx = str(context or "casual").lower()
    if ctx in ("historical", "co_trang", "cổ trang"):
        return (
            "QUY TẮC NGỮ CẢNH: CỔ TRANG (Kiếm hiệp, Cung đình, Dã sử).\n"
            "- Xưng hô: Dùng xưng hô cổ trang kiếm hiệp chuẩn mực: ta/ngươi, huynh/đệ/tỷ/muội, bệ hạ, điện hạ, công tử, tại hạ, vi thần, bổn cung, tiểu nhân, các hạ, phụ thân, mẫu thân... tùy quan hệ và thân phận.\n"
            "- Văn phong & từ ngữ: Dùng nhiều từ Hán-Việt thích hợp, câu văn trang trọng, hàm súc, hào sảng hoặc uy nghiêm mang phong vị kiếm hiệp, cổ trang."
        )
    elif ctx in ("modern", "hien_dai", "hiện đại"):
        return (
            "QUY TẮC NGỮ CẢNH: HIỆN ĐẠI (Công sở, Đô thị, Phim chính kịch, Tin tức).\n"
            "- Xưng hô: Dùng xưng hô lịch thiệp, văn minh: tôi/bạn, anh/chị/em, quý khách, ông/bà...\n"
            "- Văn phong & từ ngữ: Câu văn gãy gọn, trong sáng, từ ngữ hiện đại, chuẩn mực, không dùng từ ngữ cổ xưa hoặc tiếng lóng bỗ bã."
        )
    else:  # casual / default
        return (
            "QUY TẮC NGỮ CẢNH: ĐỜI THƯỜNG (Hội thoại thường ngày, Vlogs, Phim đời sống - Mặc định).\n"
            "- Xưng hô: Xưng hô linh hoạt, tự nhiên, gần gũi theo đời sống hàng ngày (mình/bạn, anh/em, cậu/tớ, bố/con, mày/tao nếu giận dữ hoặc bạn bè thân mật).\n"
            "- Văn phong & từ ngữ: Dùng khẩu ngữ tự nhiên, từ ngữ giao tiếp đời thực (dạ, ừ, nè, ha, á, rồi sao, đi thôi...), sống động như người Việt trò chuyện ngoài đời."
        )


class BatchResultList(list):
    """List of result items with attached token usage metadata."""

    def __init__(self, items: list[dict] | None = None, usage: dict[str, int] | None = None):
        super().__init__(items or [])
        self.usage = usage or {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}


def _call_gemini_batch_postprocess(
    batch: list[SubtitleSegment],
    fix_ocr_spelling: bool,
    translate_to: str | None,
    api_key: str,
    translation_context: TranslationContext | str = TranslationContext.CASUAL,
    preferred_model: str | None = None,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
) -> BatchResultList:
    """
    Send a batch of subtitle segments to Gemini with context window and duration constraints.
    Returns: BatchResultList containing parsed dicts and .usage token metadata.
    """
    from google import genai
    from google.genai import types
    from app.utils.gemini_client import generate_content_with_resilience

    if not batch:
        return BatchResultList([], {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0})

    client = genai.Client(api_key=api_key)
    context_rules = _get_context_instruction(translation_context)

    # Prepare batch payload with display duration constraint
    payload_items = [
        {
            "id": seg.id or f"idx_{idx}",
            "i": idx,
            "t": seg.text,
            "duration": max(0.5, round(seg.end_time - seg.start_time, 2)),
        }
        for idx, seg in enumerate(batch)
    ]

    input_payload = {
        "context_before": [t for t in (context_before or []) if t and t.strip()],
        "items_to_translate": payload_items,
        "context_after": [t for t in (context_after or []) if t and t.strip()],
    }

    task_instructions: list[str] = []
    if fix_ocr_spelling and translate_to:
        task_instructions.append(
            f"1. Sửa lỗi chính tả/dấu câu OCR ở câu gốc tiếng gốc trong 't'.\n"
            f"2. Dịch tự nhiên và chính xác sang '{translate_to}' trong trường 'tr'.\n"
            "3. Trả về đúng mảng JSON: [{\"id\": string, \"t\": string, \"tr\": string}]."
        )
    elif translate_to:
        task_instructions.append(
            f"1. Dịch từng câu phụ đề sang '{translate_to}' trong trường 'tr'.\n"
            "2. Trả về đúng mảng JSON: [{\"id\": string, \"tr\": string}]."
        )
    elif fix_ocr_spelling:
        task_instructions.append(
            "1. Sửa lỗi chính tả OCR và dấu thanh bị vỡ trong 't'.\n"
            "2. Trả về đúng mảng JSON: [{\"id\": string, \"t\": string}]."
        )

    prompt = (
        "You are an expert subtitle editor and professional video translator.\n\n"
        f"{context_rules}\n\n"
        "QUY TẮC BẮT BUỘC KHI XỬ LÝ:\n"
        + "\n".join(task_instructions)
        + "\n4. 'context_before' và 'context_after' CHỈ là ngữ cảnh tham khảo để nắm bắt mạch hội thoại và cách xưng hô. "
        "TUYỆT ĐỐI KHÔNG dịch và KHÔNG đưa chúng vào danh sách trả về.\n"
        "5. CHỈ xử lý các câu nằm trong danh sách 'items_to_translate'.\n"
        "6. Độ dài bản dịch phải vừa vặn, súc tích với 'duration' (thời lượng giây hiển thị trên màn hình). "
        "Với duration ngắn (< 1.5s), dịch ngắn gọn, cô đọng, tránh dịch dài lê thê khiến người xem không kịp đọc.\n"
        "7. Tuyệt đối không thêm dấu chấm '.' ở cuối câu phụ đề (giữ nguyên '?', '!', '...', số thập phân hoặc từ viết tắt).\n"
        "8. Kết quả phải trả về đủ đúng số lượng phần tử bằng với 'items_to_translate' và giữ nguyên trường 'id'.\n\n"
        "Input:\n"
        + json.dumps(input_payload, ensure_ascii=False)
    )

    total_usage = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}

    try:
        response = generate_content_with_resilience(
            client=client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
            preferred_model=preferred_model,
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            c = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            tot = getattr(response.usage_metadata, "total_token_count", 0) or (p + c)
            total_usage["prompt_tokens"] += p
            total_usage["candidate_tokens"] += c
            total_usage["total_tokens"] += tot

        logger.info(
            "Gemini batch call completed: prompt_tokens=%d, candidate_tokens=%d, total_tokens=%d",
            total_usage["prompt_tokens"],
            total_usage["candidate_tokens"],
            total_usage["total_tokens"],
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
            parsed = []

        if isinstance(parsed, dict):
            parsed = parsed.get("items") or parsed.get("subtitles") or [parsed]
        if not isinstance(parsed, list):
            parsed = []

    except Exception as exc:
        logger.warning("Lỗi gọi Gemini batch: %s", exc)
        parsed = []

    return BatchResultList(parsed, usage=total_usage)


async def run_llm_postprocess_pipeline(
    req: LLMJobRequest,
    job_id: str,
    emit: EventCallback,
    cancel_event: asyncio.Event | None = None,
) -> list[SubtitleSegment]:
    """
    Run high-speed concurrent batch Gemini AI spellcheck and/or bilingual translation with:
    - Translation context (historical, modern, casual)
    - Persistent SQLite caching for cost control
    - Context window and granular retries
    - Token tracking and real-time streaming
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
        context_label = {
            TranslationContext.HISTORICAL: "Cổ trang",
            TranslationContext.MODERN: "Hiện đại",
            TranslationContext.CASUAL: "Đời thường",
        }.get(req.translation_context, "Đời thường")

        action_desc = []
        if req.fix_ocr_spelling:
            action_desc.append("Sửa lỗi chính tả")
        if req.translate_to:
            action_desc.append(f"Dịch sang {req.translate_to} [{context_label}]")
        action_title = " + ".join(action_desc) or "Xử lý AI"

        await emit(
            WebSocketEvent(
                event="job_started",
                project_id=req.project_id,
                job_id=job_id,
                progress=1.0,
                message=f"Bắt đầu {action_title} ({len(segments)} dòng phụ đề)...",
            )
        )

        all_updated_segments: list[SubtitleSegment] = list(segments)
        context_str = req.translation_context.value if hasattr(req.translation_context, "value") else str(req.translation_context)

        # 1. Cost Control & Cache lookup
        cached_applied_count = 0
        cache_updates: list[dict] = []
        indices_to_call_api: list[int] = []

        if req.translate_to:
            # Query existing translation_cache for all segments
            cached_translations = await db.get_cached_translations(
                target_lang=req.translate_to,
                context_type=context_str,
                source_texts=[s.text for s in segments],
            )

            for idx, seg in enumerate(segments):
                clean_src = seg.text.strip()
                # Skip if already translated and not forcing re-translation
                if not req.force_retranslate and seg.translated_text and seg.translated_text.strip():
                    cached_applied_count += 1
                    continue

                # Check if present in translation_cache
                if not req.force_retranslate and clean_src in cached_translations:
                    cached_trans = remove_trailing_period(cached_translations[clean_src])
                    seg.translated_text = cached_trans
                    all_updated_segments[idx] = seg
                    cache_updates.append({"id": seg.id, "translated_text": cached_trans})
                    cached_applied_count += 1
                else:
                    indices_to_call_api.append(idx)
        else:
            # Spellcheck only: process all
            indices_to_call_api = list(range(len(segments)))

        # If cache updates were found, save them atomically and broadcast
        if cache_updates:
            await db.batch_update_segments(req.project_id, cache_updates)
            for upd in cache_updates:
                seg_match = next((s for s in all_updated_segments if s.id == upd["id"]), None)
                if seg_match:
                    await emit(
                        WebSocketEvent(
                            event="segment_updated",
                            project_id=req.project_id,
                            job_id=job_id,
                            progress=5.0,
                            segment=seg_match,
                        )
                    )

        # Check if all items were covered by cache / existing translations
        if not indices_to_call_api:
            elapsed = round(time.perf_counter() - start_wall, 2)
            await db.update_project_status(req.project_id, status=JobStatus.COMPLETED, progress=100.0)
            await emit(
                WebSocketEvent(
                    event="job_completed",
                    project_id=req.project_id,
                    job_id=job_id,
                    progress=100.0,
                    message=f"Đã cập nhật toàn bộ {len(segments)} phụ đề từ bộ nhớ đệm (Cache) trong {elapsed}s, không tốn thêm token.",
                    extra={
                        "segment_count": len(segments),
                        "cached_count": cached_applied_count,
                        "tokens_used": 0,
                        "elapsed_sec": elapsed,
                    },
                )
            )
            return all_updated_segments

        # 2. Divide remaining items into batches of size req.batch_size (default 50)
        batch_size = max(10, int(req.batch_size or 50))
        batch_index_groups = [
            indices_to_call_api[i : i + batch_size]
            for i in range(0, len(indices_to_call_api), batch_size)
        ]

        total_to_process = len(indices_to_call_api)
        completed_count = cached_applied_count
        progress_lock = asyncio.Lock()
        total_tokens_used = 0
        total_prompt_tokens = 0
        total_candidate_tokens = 0
        sem = asyncio.Semaphore(3)  # Respect Gemini rate limits

        async def process_batch_indices(idx_group: list[int]):
            nonlocal completed_count, total_tokens_used, total_prompt_tokens, total_candidate_tokens
            async with sem:
                if cancel_ev.is_set():
                    return

                batch_segs = [segments[i] for i in idx_group]

                # Context window: extract up to 3 preceding and 3 succeeding lines from full timeline
                min_idx = idx_group[0]
                max_idx = idx_group[-1]
                ctx_before = [segments[i].text for i in range(max(0, min_idx - 3), min_idx)]
                ctx_after = [segments[i].text for i in range(max_idx + 1, min(len(segments), max_idx + 4))]

                try:
                    raw_results = await asyncio.to_thread(
                        _call_gemini_batch_postprocess,
                        batch=batch_segs,
                        fix_ocr_spelling=req.fix_ocr_spelling,
                        translate_to=req.translate_to,
                        api_key=api_key,
                        translation_context=req.translation_context,
                        preferred_model=req.model,
                        context_before=ctx_before,
                        context_after=ctx_after,
                    )
                except TypeError:
                    # Backward-compatibility for unit test mocks with 4 arguments
                    raw_results = await asyncio.to_thread(
                        _call_gemini_batch_postprocess,
                        batch_segs,
                        req.fix_ocr_spelling,
                        req.translate_to,
                        api_key,
                    )
                usage = getattr(raw_results, "usage", {}) if hasattr(raw_results, "usage") else {}

                # Index results by id and integer index i
                by_id: dict[str, dict] = {}
                by_i: dict[int, dict] = {}
                for idx, item in enumerate(raw_results):
                    if isinstance(item, dict):
                        if "id" in item and item["id"]:
                            by_id[str(item["id"])] = item
                        if "i" in item:
                            try:
                                by_i[int(item["i"])] = item
                            except (ValueError, TypeError):
                                pass

                # Check if any segments are missing from results
                missing_in_batch: list[SubtitleSegment] = []
                for pos, seg in enumerate(batch_segs):
                    if by_id:
                        if str(seg.id) not in by_id:
                            missing_in_batch.append(seg)
                    elif by_i:
                        if pos not in by_i:
                            missing_in_batch.append(seg)
                    else:
                        if pos >= len(raw_results):
                            missing_in_batch.append(seg)

                # Granular retry: only retry missing lines
                if missing_in_batch:
                    logger.warning(
                        "Batch thiếu %d/%d dòng. Thử lại riêng các dòng lỗi...",
                        len(missing_in_batch),
                        len(batch_segs),
                    )
                    try:
                        retry_results = await asyncio.to_thread(
                            _call_gemini_batch_postprocess,
                            batch=missing_in_batch,
                            fix_ocr_spelling=req.fix_ocr_spelling,
                            translate_to=req.translate_to,
                            api_key=api_key,
                            translation_context=req.translation_context,
                            preferred_model=req.model,
                            context_before=ctx_before,
                            context_after=ctx_after,
                        )
                        retry_usage = getattr(retry_results, "usage", {}) if hasattr(retry_results, "usage") else {}
                        usage["total_tokens"] = usage.get("total_tokens", 0) + retry_usage.get("total_tokens", 0)
                        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + retry_usage.get("prompt_tokens", 0)
                        usage["candidate_tokens"] = usage.get("candidate_tokens", 0) + retry_usage.get("candidate_tokens", 0)

                        for r_idx, r_item in enumerate(retry_results):
                            if isinstance(r_item, dict):
                                if "id" in r_item and r_item["id"]:
                                    by_id[str(r_item["id"])] = r_item
                                elif r_idx < len(missing_in_batch):
                                    by_id[missing_in_batch[r_idx].id] = r_item
                    except Exception as retry_err:
                        logger.warning("Thử lại các dòng lỗi thất bại: %s", retry_err)

                async with progress_lock:
                    total_tokens_used += usage.get("total_tokens", 0)
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_candidate_tokens += usage.get("candidate_tokens", 0)

                batch_db_updates = []
                new_cache_entries: list[tuple[str, str, str, str]] = []
                batch_result_segments: list[SubtitleSegment] = []

                for pos, orig_idx in enumerate(idx_group):
                    seg = segments[orig_idx]
                    res_item = by_id.get(str(seg.id)) or by_i.get(pos) or (raw_results[pos] if pos < len(raw_results) else None)

                    if res_item and isinstance(res_item, dict):
                        new_text = res_item.get("t") or res_item.get("text") or seg.text
                        new_trans = (
                            res_item.get("tr")
                            or res_item.get("translated_text")
                            or res_item.get("translation")
                        )
                        if new_trans is not None:
                            new_trans = str(new_trans).strip() or None

                        upd_text = (
                            remove_trailing_period(str(new_text).strip())
                            if req.fix_ocr_spelling
                            else seg.text
                        )
                        upd_trans = (
                            remove_trailing_period(new_trans)
                            if (req.translate_to and new_trans is not None)
                            else (new_trans if req.translate_to else seg.translated_text)
                        )

                        batch_db_updates.append({
                            "id": seg.id,
                            "text": upd_text,
                            "translated_text": upd_trans,
                        })
                        seg.text = upd_text
                        seg.translated_text = upd_trans

                        # Prepare entry for persistent translation cache
                        if req.translate_to and upd_trans:
                            new_cache_entries.append((
                                seg.text.strip(),
                                req.translate_to.strip(),
                                context_str,
                                upd_trans,
                            ))

                    batch_result_segments.append(seg)
                    all_updated_segments[orig_idx] = seg

                # Atomic updates to database and translation cache
                if batch_db_updates:
                    await db.batch_update_segments(req.project_id, batch_db_updates)

                if new_cache_entries:
                    await db.save_cached_translations(new_cache_entries)

                # Update progress and broadcast updated segments
                async with progress_lock:
                    completed_count += len(batch_segs)
                    pct = min(99.0, (completed_count / len(segments)) * 100.0)

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

        # Run batches concurrently
        tasks = [process_batch_indices(group) for group in batch_index_groups]
        await asyncio.gather(*tasks)

        if cancel_ev.is_set():
            await db.update_project_status(req.project_id, status=JobStatus.CANCELLED)
            await emit(
                WebSocketEvent(
                    event="job_cancelled",
                    project_id=req.project_id,
                    job_id=job_id,
                    message="Đã hủy tác vụ AI dịch thuật.",
                )
            )
            return all_updated_segments

        elapsed = round(time.perf_counter() - start_wall, 2)
        await db.update_project_status(req.project_id, status=JobStatus.COMPLETED, progress=100.0)

        logger.info(
            "Hoàn tất dịch project %s: %d câu (tổng token: %d, prompt: %d, candidate: %d, cache: %d)",
            req.project_id,
            len(all_updated_segments),
            total_tokens_used,
            total_prompt_tokens,
            total_candidate_tokens,
            cached_applied_count,
        )

        success_msg = (
            f"Hoàn tất dịch AI {len(all_updated_segments)} câu phụ đề "
            f"({len(indices_to_call_api)} câu mới, {cached_applied_count} từ cache) "
            f"hết {total_tokens_used:,} tokens trong {elapsed}s."
        )

        await emit(
            WebSocketEvent(
                event="job_completed",
                project_id=req.project_id,
                job_id=job_id,
                progress=100.0,
                message=success_msg,
                extra={
                    "segment_count": len(all_updated_segments),
                    "api_called_count": len(indices_to_call_api),
                    "cached_count": cached_applied_count,
                    "tokens_used": total_tokens_used,
                    "prompt_tokens": total_prompt_tokens,
                    "candidate_tokens": total_candidate_tokens,
                    "elapsed_sec": elapsed,
                },
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
                message=f"Lỗi AI dịch thuật: {exc}",
            )
        )
        raise


async def estimate_llm_cost(req: LLMEstimateRequest) -> LLMEstimateResponse:
    """Estimate token usage and cost for an LLM translation/polish job before execution."""
    project = await db.get_project(req.project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")

    segments = list(project.segments)
    total_segments = len(segments)
    if total_segments == 0:
        return LLMEstimateResponse(
            total_segments=0,
            cached_segments=0,
            segments_to_translate=0,
            estimated_prompt_tokens=0,
            estimated_completion_tokens=0,
            estimated_total_tokens=0,
            estimated_cost_usd=0.0,
        )

    cached_count = 0
    needing_trans: list[SubtitleSegment] = []
    context_str = req.translation_context.value if hasattr(req.translation_context, "value") else str(req.translation_context)

    for seg in segments:
        if not req.force_retranslate and seg.translated_text and seg.translated_text.strip():
            cached_count += 1
        else:
            needing_trans.append(seg)

    if req.translate_to and needing_trans:
        cached_map = await db.get_cached_translations(
            target_lang=req.translate_to,
            context_type=context_str,
            source_texts=[s.text for s in needing_trans],
        )
        final_needing = []
        for s in needing_trans:
            if not req.force_retranslate and s.text.strip() in cached_map:
                cached_count += 1
            else:
                final_needing.append(s)
        needing_trans = final_needing

    to_translate_count = len(needing_trans)
    total_words = sum(len(s.text.split()) for s in needing_trans) if needing_trans else 0
    batches_count = max(1, (to_translate_count + 49) // 50) if to_translate_count > 0 else 0

    estimated_prompt_tokens = int(total_words * 1.35) + (batches_count * 380) if to_translate_count > 0 else 0
    estimated_completion_tokens = int(total_words * 1.4) if to_translate_count > 0 else 0
    estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

    # Standard Gemini 3.8 Flash rates: ~$0.075 / 1M prompt, ~$0.30 / 1M candidate tokens
    cost_usd = (estimated_prompt_tokens * 0.000000075) + (estimated_completion_tokens * 0.00000030)

    return LLMEstimateResponse(
        total_segments=total_segments,
        cached_segments=cached_count,
        segments_to_translate=to_translate_count,
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_completion_tokens,
        estimated_total_tokens=estimated_total_tokens,
        estimated_cost_usd=round(cost_usd, 6),
    )
