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
import re
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
from app.utils.cjk_normalizer import convert_t2s
from app.utils.subtitle_formatter import remove_trailing_period

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]

CJK_IDEOGRAPH_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)

TARGET_LANG_NAMES: dict[str, str] = {
    "vi": "Tiếng Việt (Vietnamese)",
    "en": "Tiếng Anh (English)",
    "zh": "Tiếng Trung (Chinese)",
    "ja": "Tiếng Nhật (Japanese)",
    "ko": "Tiếng Hàn (Korean)",
    "fr": "Tiếng Pháp (French)",
    "de": "Tiếng Đức (German)",
    "es": "Tiếng Tây Ban Nha (Spanish)",
    "ru": "Tiếng Nga (Russian)",
    "th": "Tiếng Thái (Thai)",
}


def _resolve_target_lang_name(translate_to: str | None) -> str:
    """Resolve short language code (e.g. 'vi', 'en') to an explicit human-readable language name."""
    code = (translate_to or "vi").strip()
    return TARGET_LANG_NAMES.get(code.lower(), code)


def _is_valid_translation(
    source_text: str | None,
    translated_text: str | None,
    target_lang: str | None,
) -> bool:
    """
    Validate that a translated subtitle line is non-empty and actually translated into target_lang
    (e.g. rejects untranslated Chinese/CJK output when target_lang is Vietnamese or English).
    """
    if translated_text is None:
        return False
    t_str = str(translated_text).strip()
    if not t_str:
        return False
    s_str = (source_text or "").strip()
    lang_code = (target_lang or "vi").strip().lower()
    is_cjk_target = any(
        k in lang_code
        for k in ("zh", "ch", "ja", "ko", "chinese", "japanese", "korean", "trung", "nhật", "hàn")
    )
    if not is_cjk_target and CJK_IDEOGRAPH_RE.search(s_str):
        if CJK_IDEOGRAPH_RE.search(t_str):
            return False
    return True


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
    effective_target_lang = translate_to or ("vi" if fix_ocr_spelling else None)
    target_lang_label = _resolve_target_lang_name(effective_target_lang)

    # Prepare batch payload with display duration constraint
    payload_items = []
    for idx, seg in enumerate(batch):
        item_dict: dict[str, object] = {
            "id": seg.id or f"idx_{idx}",
            "i": idx,
            "t": seg.text,
            "duration": max(0.5, round(seg.end_time - seg.start_time, 2)),
        }
        if fix_ocr_spelling and seg.translated_text:
            item_dict["tr_draft"] = seg.translated_text
        payload_items.append(item_dict)

    input_payload = {
        "context_before": [t for t in (context_before or []) if t and t.strip()],
        "items_to_translate": payload_items,
        "context_after": [t for t in (context_after or []) if t and t.strip()],
    }

    task_instructions: list[str] = []
    if fix_ocr_spelling:
        task_instructions.append(
            "1. Trường 't' là văn bản gốc dùng để đối chiếu ngữ nghĩa. TUYỆT ĐỐI giữ nguyên bản gốc, KHÔNG dịch đè lên bản gốc.\n"
            f"2. Trau chuốt, sửa lỗi diễn đạt/chính tả và cải thiện bản dịch trong 'tr_draft' (hoặc dịch mới từ 't' nếu chưa có 'tr_draft' hoặc còn sót chữ gốc) sang {target_lang_label} thật tự nhiên, mượt mà, chuẩn ngữ cảnh và đặt kết quả vào trường 'tr'.\n"
            "3. Trả về đúng mảng JSON: [{\"id\": string, \"tr\": string}]."
        )
    elif translate_to:
        task_instructions.append(
            f"1. Dịch đầy đủ 100% từng câu phụ đề trong 't' sang {target_lang_label} trong trường 'tr'. TUYỆT ĐỐI KHÔNG bỏ sót dòng nào và KHÔNG để sót chữ ngôn ngữ gốc chưa dịch trong 'tr'.\n"
            "2. Trả về đúng mảng JSON: [{\"id\": string, \"tr\": string}]."
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
        effective_translate_to = req.translate_to or ("vi" if req.fix_ocr_spelling else None)
        if req.fix_ocr_spelling:
            action_desc.append(f"AI Polish bản dịch ({effective_translate_to}) [{context_label}]")
        elif req.translate_to:
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

        req_model = req.model or "default"
        req_pver = req.prompt_version or "v1.0"

        if req.translate_to:
            # Query existing translation_cache for all segments using normalized text & 5-tuple composite key
            query_sources = [
                (s.normalized_text or convert_t2s(s.text or "")).strip()
                for s in segments
            ]
            cached_translations = await db.get_cached_translations(
                target_lang=req.translate_to,
                context_type=context_str,
                source_texts=query_sources,
                model_name=req_model,
                prompt_version=req_pver,
            )

            for idx, seg in enumerate(segments):
                clean_src = (seg.normalized_text or convert_t2s(seg.text or "")).strip()
                # Skip if already translated with a valid translation and not forcing re-translation
                if (
                    not req.force_retranslate
                    and _is_valid_translation(seg.text, seg.translated_text, req.translate_to)
                ):
                    cached_applied_count += 1
                    continue

                # Check if present in translation_cache with a valid translation
                if (
                    not req.force_retranslate
                    and clean_src in cached_translations
                    and _is_valid_translation(seg.text, cached_translations[clean_src], req.translate_to)
                ):
                    cached_trans = remove_trailing_period(cached_translations[clean_src])
                    seg.translated_text = cached_trans
                    all_updated_segments[idx] = seg
                    cache_updates.append({"id": seg.id, "translated_text": cached_trans})
                    cached_applied_count += 1
                else:
                    indices_to_call_api.append(idx)
        else:
            # AI Polish without explicit translate_to: process all segments
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

        # 2. Divide remaining items into batches of size min(req.batch_size, 30) to prevent LLM drift/truncation
        batch_size = max(10, min(35, int(req.batch_size or 30))) if not req.batch_size or req.batch_size == 50 else max(10, int(req.batch_size))
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

        def _extract_item_trans(
            res_item: dict | None, orig_text: str | None = None
        ) -> str | None:
            if not res_item or not isinstance(res_item, dict):
                return None
            cand = (
                res_item.get("tr")
                or res_item.get("translated_text")
                or res_item.get("translation")
            )
            if cand is None and req.fix_ocr_spelling:
                fallback_t = res_item.get("t") or res_item.get("text")
                if (
                    fallback_t is not None
                    and str(fallback_t).strip()
                    and str(fallback_t).strip() != (orig_text or "").strip()
                ):
                    cand = fallback_t
            if cand is not None:
                cand = str(cand).strip() or None
            return cand

        def _match_batch_item(
            pos: int,
            seg: SubtitleSegment,
            by_id: dict[str, dict],
            by_i: dict[int, dict],
            raw_list: list,
            batch_id_set: set[str],
        ) -> dict | None:
            seg_id_str = str(seg.id)
            if seg_id_str in by_id:
                return by_id[seg_id_str]
            if f"idx_{pos}" in by_id:
                return by_id[f"idx_{pos}"]
            if str(pos) in by_id and str(pos) not in batch_id_set:
                return by_id[str(pos)]
            if pos in by_i:
                return by_i[pos]
            if pos < len(raw_list) and isinstance(raw_list[pos], dict):
                cand_item = raw_list[pos]
                cand_id = str(cand_item.get("id") or "")
                cand_i = cand_item.get("i")
                # Only use positional fallback if the item does not explicitly belong to another segment
                if cand_id and cand_id in batch_id_set and cand_id != seg_id_str:
                    return None
                if cand_i is not None:
                    try:
                        if int(cand_i) != pos:
                            return None
                    except (ValueError, TypeError):
                        pass
                return cand_item
            return None

        async def process_batch_indices(idx_group: list[int]):
            nonlocal completed_count, total_tokens_used, total_prompt_tokens, total_candidate_tokens
            async with sem:
                if cancel_ev.is_set():
                    return

                batch_segs = [segments[i] for i in idx_group]
                batch_id_set = {str(s.id) for s in batch_segs if s.id}

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
                        translate_to=effective_translate_to,
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
                        effective_translate_to,
                        api_key,
                    )
                usage = getattr(raw_results, "usage", {}) if hasattr(raw_results, "usage") else {}

                # Index results by id and integer index i
                by_id: dict[str, dict] = {}
                by_i: dict[int, dict] = {}
                for idx, item in enumerate(raw_results):
                    if isinstance(item, dict):
                        if "id" in item and item["id"] is not None and str(item["id"]) != "":
                            by_id[str(item["id"])] = item
                        if "i" in item and item["i"] is not None:
                            try:
                                by_i[int(item["i"])] = item
                            except (ValueError, TypeError):
                                pass

                # Check if any segments are missing OR have invalid/untranslated output
                missing_in_batch: list[SubtitleSegment] = []
                for pos, seg in enumerate(batch_segs):
                    matched_item = _match_batch_item(
                        pos, seg, by_id, by_i, raw_results, batch_id_set
                    )
                    cand_tr = _extract_item_trans(matched_item, seg.text)
                    if matched_item is None or (
                        effective_translate_to
                        and not _is_valid_translation(seg.text, cand_tr, effective_translate_to)
                    ):
                        missing_in_batch.append(seg)

                # Granular retry: retry missing or untranslated lines in small sub-batches (<= 15 items)
                if missing_in_batch:
                    logger.warning(
                        "Batch thiếu hoặc chưa dịch %d/%d dòng. Thử lại riêng các dòng lỗi...",
                        len(missing_in_batch),
                        len(batch_segs),
                    )
                    retry_chunk_size = 15
                    for r_start in range(0, len(missing_in_batch), retry_chunk_size):
                        retry_chunk = missing_in_batch[r_start : r_start + retry_chunk_size]
                        retry_chunk_ids = {str(s.id): s for s in retry_chunk if s.id}
                        try:
                            try:
                                retry_results = await asyncio.to_thread(
                                    _call_gemini_batch_postprocess,
                                    batch=retry_chunk,
                                    fix_ocr_spelling=req.fix_ocr_spelling,
                                    translate_to=effective_translate_to,
                                    api_key=api_key,
                                    translation_context=req.translation_context,
                                    preferred_model=req.model,
                                    context_before=ctx_before,
                                    context_after=ctx_after,
                                )
                            except TypeError:
                                retry_results = await asyncio.to_thread(
                                    _call_gemini_batch_postprocess,
                                    retry_chunk,
                                    req.fix_ocr_spelling,
                                    effective_translate_to,
                                    api_key,
                                )
                            retry_usage = getattr(retry_results, "usage", {}) if hasattr(retry_results, "usage") else {}
                            usage["total_tokens"] = usage.get("total_tokens", 0) + retry_usage.get("total_tokens", 0)
                            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + retry_usage.get("prompt_tokens", 0)
                            usage["candidate_tokens"] = usage.get("candidate_tokens", 0) + retry_usage.get("candidate_tokens", 0)

                            for r_idx, r_item in enumerate(retry_results):
                                if not isinstance(r_item, dict):
                                    continue
                                target_seg_id: str | None = None
                                raw_rid = str(r_item.get("id")) if r_item.get("id") is not None else ""
                                if raw_rid in retry_chunk_ids:
                                    target_seg_id = raw_rid
                                elif raw_rid.startswith("idx_"):
                                    try:
                                        parsed_i = int(raw_rid.split("_", 1)[1])
                                        if 0 <= parsed_i < len(retry_chunk):
                                            target_seg_id = str(retry_chunk[parsed_i].id)
                                    except (ValueError, TypeError):
                                        pass
                                elif raw_rid.isdigit() and int(raw_rid) < len(retry_chunk):
                                    target_seg_id = str(retry_chunk[int(raw_rid)].id)
                                elif "i" in r_item and r_item["i"] is not None:
                                    try:
                                        parsed_i = int(r_item["i"])
                                        if 0 <= parsed_i < len(retry_chunk):
                                            target_seg_id = str(retry_chunk[parsed_i].id)
                                    except (ValueError, TypeError):
                                        pass
                                if target_seg_id is None and r_idx < len(retry_chunk):
                                    target_seg_id = str(retry_chunk[r_idx].id)
                                if target_seg_id:
                                    by_id[target_seg_id] = r_item
                        except Exception as retry_err:
                            logger.warning("Thử lại các dòng lỗi thất bại: %s", retry_err)

                async with progress_lock:
                    total_tokens_used += usage.get("total_tokens", 0)
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_candidate_tokens += usage.get("candidate_tokens", 0)

                batch_db_updates = []
                new_cache_entries: list[tuple[str, str, str, str, str, str]] = []
                batch_result_segments: list[SubtitleSegment] = []

                for pos, orig_idx in enumerate(idx_group):
                    seg = segments[orig_idx]
                    res_item = _match_batch_item(
                        pos, seg, by_id, by_i, raw_results, batch_id_set
                    )

                    if res_item and isinstance(res_item, dict):
                        new_trans = _extract_item_trans(res_item, seg.text)

                        # Requirement 3: Strictly preserve original text (seg.text / seg.original_text);
                        # AI Polish and AI Translate only update translated_text.
                        upd_text = seg.text
                        if effective_translate_to and new_trans is not None:
                            cleaned_cand = remove_trailing_period(new_trans)
                            if _is_valid_translation(seg.text, cleaned_cand, effective_translate_to):
                                upd_trans = cleaned_cand
                            elif _is_valid_translation(
                                seg.text, seg.translated_text, effective_translate_to
                            ):
                                # Preserve existing valid draft translation if retry still returned CJK drift
                                upd_trans = seg.translated_text
                            else:
                                upd_trans = cleaned_cand
                        else:
                            upd_trans = seg.translated_text

                        batch_db_updates.append({
                            "id": seg.id,
                            "text": upd_text,
                            "translated_text": upd_trans,
                        })
                        seg.text = upd_text
                        seg.translated_text = upd_trans

                        # Prepare entry for persistent translation cache using 5-tuple composite key
                        # Only cache valid translations (never cache untranslated CJK drift)
                        if (
                            effective_translate_to
                            and upd_trans
                            and _is_valid_translation(seg.text, upd_trans, effective_translate_to)
                        ):
                            cache_src = (seg.normalized_text or convert_t2s(seg.text or "")).strip()
                            new_cache_entries.append((
                                cache_src,
                                effective_translate_to.strip(),
                                context_str,
                                req_model,
                                req_pver,
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
        if not req.force_retranslate and _is_valid_translation(
            seg.text, seg.translated_text, req.translate_to
        ):
            cached_count += 1
        else:
            needing_trans.append(seg)

    if req.translate_to and needing_trans:
        query_sources = [
            (s.normalized_text or convert_t2s(s.text or "")).strip()
            for s in needing_trans
        ]
        cached_map = await db.get_cached_translations(
            target_lang=req.translate_to,
            context_type=context_str,
            source_texts=query_sources,
        )
        final_needing = []
        for s in needing_trans:
            src_k = (s.normalized_text or convert_t2s(s.text or "")).strip()
            if (
                not req.force_retranslate
                and src_k in cached_map
                and _is_valid_translation(s.text, cached_map[src_k], req.translate_to)
            ):
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
