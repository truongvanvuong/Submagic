"""
Phase 2: Speech-to-Text (STT) Pipeline.
Features:
- FFmpeg / PyAV 16kHz vocal-normalized audio preprocessing
- Faster-Whisper with automatic CUDA FP16 -> CPU INT8 fallback
- Silero VAD filtering & anti-hallucination guards
- Smart Word-Level Subtitle Segmenter & balanced multi-line wrapper
- Optional Cloud STT via Gemini 2.5 Flash API
- Real-time SQLite auto-save and WebSocket progress/segment streaming
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from app.config import settings
from app.database import db
from app.schemas import (
    ExportFormat,
    ExtractionMode,
    JobStatus,
    STTJobRequest,
    SubtitleSegment,
    WebSocketEvent,
    WordTimestamp,
)
from app.services.ffmpeg_service import extract_audio_for_stt, probe_video, resolve_media_path
from app.services.hardware import resolve_stt_device_and_compute
from app.utils.subtitle_formatter import format_subtitles

logger = logging.getLogger(__name__)

EventCallback = Callable[[WebSocketEvent], Awaitable[None]]

# Sentence and clause punctuation sets (Latin, Vietnamese, CJK)
SENTENCE_END_PUNCT = (".", "?", "!", "…", "。", "？", "！", ";", "；")
CLAUSE_END_PUNCT = (",", "，", "、", ":", "：", "-")

# Common Whisper silence/BGM hallucination patterns when confidence is low
HALLUCINATION_PATTERNS = [
    re.compile(r"^(hãy subscribe|đăng ký kênh|cảm ơn các bạn đã xem|thank you for watching|subtitles by|amara\.org)", re.I),
]


class WhisperModelCache:
    """In-memory cache for loaded Faster-Whisper models with CUDA-to-CPU fallback."""

    def __init__(self) -> None:
        self._model = None
        self._loaded_key: tuple[str, str, str] | None = None

    def get_model(self, model_size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        target_key = (model_size, device, compute_type)
        if self._model is not None and self._loaded_key == target_key:
            return self._model, device, compute_type

        download_root = str(settings.models_dir / "whisper")
        Path(download_root).mkdir(parents=True, exist_ok=True)

        # Helper to load model offline-first from local cache to avoid HuggingFace network latency
        def _load_whisper(m_size: str, dev: str, comp: str) -> WhisperModel:
            try:
                # 1. Try loading directly from local cache (offline, fast, no network requests)
                return WhisperModel(
                    m_size,
                    device=dev,
                    compute_type=comp,
                    download_root=download_root,
                    local_files_only=True,
                )
            except Exception as cache_err:
                # 2. If not found locally, download from HuggingFace
                logger.info(
                    "Model '%s' not cached locally (%s). Downloading from HuggingFace...",
                    m_size,
                    cache_err,
                )
                return WhisperModel(
                    m_size,
                    device=dev,
                    compute_type=comp,
                    download_root=download_root,
                    local_files_only=False,
                )

        try:
            logger.info(
                "Loading Faster-Whisper model '%s' on device='%s' (compute_type='%s')...",
                model_size,
                device,
                compute_type,
            )
            self._model = _load_whisper(model_size, device, compute_type)
            self._loaded_key = target_key
            return self._model, device, compute_type
        except Exception as exc:
            if device == "cuda":
                logger.warning(
                    "CUDA initialization failed (%s). Falling back to CPU with int8.",
                    exc,
                )
                fallback_device = "cpu"
                fallback_compute = "int8"
                fallback_key = (model_size, fallback_device, fallback_compute)
                self._model = _load_whisper(model_size, fallback_device, fallback_compute)
                self._loaded_key = fallback_key
                return self._model, fallback_device, fallback_compute
            raise


model_cache = WhisperModelCache()


def wrap_subtitle_lines(
    text: str, max_chars_per_line: int = 42, max_lines: int = 2
) -> str:
    """
    Balance a subtitle string across up to `max_lines` lines so no single line
    is awkwardly long, breaking at word boundaries near the midpoint.
    """
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned or max_lines <= 1 or len(cleaned) <= max_chars_per_line:
        return cleaned

    # Split into balanced 2 lines near the middle space
    mid = len(cleaned) // 2
    left_space = cleaned.rfind(" ", 0, mid + 1)
    right_space = cleaned.find(" ", mid)

    if left_space == -1 and right_space == -1:
        # CJK text without spaces: split directly at midpoint
        return f"{cleaned[:mid]}\n{cleaned[mid:]}"

    if left_space == -1:
        split_idx = right_space
    elif right_space == -1:
        split_idx = left_space
    else:
        split_idx = (
            left_space if (mid - left_space) <= (right_space - mid) else right_space
        )

    line1 = cleaned[:split_idx].strip()
    line2 = cleaned[split_idx:].strip()
    if not line1 or not line2:
        return cleaned
    return f"{line1}\n{line2}"


def is_likely_hallucination(text: str, avg_prob: float) -> bool:
    """Detect repetitive token loops or low-confidence silence hallucinations."""
    cleaned = text.strip()
    if not cleaned:
        return True

    # Check if a single character or word repeats > 6 times consecutively
    words = cleaned.split()
    if len(words) >= 6 and len(set(w.lower() for w in words)) == 1:
        return True

    if avg_prob < 0.35:
        for pattern in HALLUCINATION_PATTERNS:
            if pattern.search(cleaned):
                return True
    return False


class SmartWordSegmenter:
    """
    Splits raw word-level timestamps from Whisper into broadcast-grade subtitle segments
    based on punctuation, natural speech pauses, and character limits.
    """

    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines_per_segment: int = 2,
        max_pause_sec: float = 0.45,
        min_segment_chars: int = 10,
    ) -> None:
        self.max_chars_per_line = max(15, max_chars_per_line)
        self.max_lines_per_segment = max(1, max_lines_per_segment)
        self.max_block_chars = self.max_chars_per_line * self.max_lines_per_segment
        self.max_pause_sec = max(0.15, max_pause_sec)
        self.min_segment_chars = min_segment_chars

    def segment_words(
        self,
        words: list[WordTimestamp],
        project_id: str,
        start_index: int = 1,
    ) -> list[SubtitleSegment]:
        if not words:
            return []

        segments: list[SubtitleSegment] = []
        current_words: list[WordTimestamp] = []
        current_index = start_index

        def flush_current() -> None:
            nonlocal current_words, current_index
            if not current_words:
                return

            raw_text = "".join(w.word for w in current_words).strip()
            raw_text = re.sub(r"\s+", " ", raw_text)
            if not raw_text:
                current_words = []
                return

            avg_prob = sum(w.probability for w in current_words) / len(current_words)
            if is_likely_hallucination(raw_text, avg_prob):
                current_words = []
                return

            formatted_text = wrap_subtitle_lines(
                raw_text,
                max_chars_per_line=self.max_chars_per_line,
                max_lines=self.max_lines_per_segment,
            )
            start_t = max(0.0, current_words[0].start)
            end_t = max(start_t + 0.20, current_words[-1].end)

            segments.append(
                SubtitleSegment(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    index=current_index,
                    start_time=round(start_t, 3),
                    end_time=round(end_t, 3),
                    text=formatted_text,
                    confidence=round(avg_prob, 4),
                    source="stt",
                    words=list(current_words),
                )
            )
            current_index += 1
            current_words = []

        for word in words:
            w_str = word.word
            if not w_str or not w_str.strip():
                continue

            if current_words:
                prev_word = current_words[-1]
                pause_gap = max(0.0, word.start - prev_word.end)
                current_text = re.sub(
                    r"\s+", " ", "".join(w.word for w in current_words).strip()
                )
                candidate_text = re.sub(
                    r"\s+", " ", f"{current_text}{w_str}".strip()
                )

                # Rule 1: Hard character limit reached
                if len(candidate_text) > self.max_block_chars:
                    flush_current()

                # Rule 2: Natural acoustic pause after a reasonable chunk
                elif (
                    pause_gap >= self.max_pause_sec
                    and len(current_text) >= self.min_segment_chars
                ):
                    flush_current()

            current_words.append(word)
            updated_text = re.sub(
                r"\s+", " ", "".join(w.word for w in current_words).strip()
            )

            # Rule 3: Sentence-terminal punctuation
            if updated_text.endswith(SENTENCE_END_PUNCT) and len(updated_text) >= 4:
                flush_current()

            # Rule 4: Clause punctuation when already past ~60% of a single line
            elif (
                updated_text.endswith(CLAUSE_END_PUNCT)
                and len(updated_text) >= int(self.max_chars_per_line * 0.65)
            ):
                flush_current()

        flush_current()
        return segments


async def _run_cloud_gemini_stt(
    audio_path: Path,
    duration_sec: float,
    req: STTJobRequest,
    job_id: str,
    emit: EventCallback,
    cancel_event: asyncio.Event,
) -> list[SubtitleSegment]:
    """Transcribe audio using Gemini 2.5 Flash API when Cloud mode is selected."""
    from google import genai
    from google.genai import types

    api_key = req.gemini_api_key or settings.gemini_api_key
    if not api_key:
        raise ValueError(
            "Thiếu Gemini API Key. Vui lòng nhập API Key hoặc chuyển sang chế độ Local Faster-Whisper."
        )

    await emit(
        WebSocketEvent(
            event="job_progress",
            project_id=req.project_id,
            job_id=job_id,
            progress=15.0,
            message="Đang tải file âm thanh lên Gemini Cloud STT...",
        )
    )

    def _sync_gemini_call() -> list[dict]:
        client = genai.Client(api_key=api_key)
        uploaded_file = client.files.upload(file=str(audio_path))
        try:
            lang_hint = f"Target language: {req.language}." if req.language else "Detect spoken language automatically."
            prompt = (
                f"Transcribe all speech in this audio accurately with precise timestamps in seconds. "
                f"{lang_hint} Split segments naturally so each subtitle is at most "
                f"{req.max_chars_per_line * req.max_lines_per_segment} characters. "
                f"Return ONLY a JSON array of objects with keys: start_time (float), end_time (float), text (string)."
            )
            from app.utils.gemini_client import generate_content_with_resilience
            response = generate_content_with_resilience(
                client=client,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            return json.loads(response.text or "[]")
        finally:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    raw_items = await asyncio.to_thread(_sync_gemini_call)
    segments: list[SubtitleSegment] = []

    for idx, item in enumerate(raw_items, start=1):
        if cancel_event.is_set():
            break
        text = wrap_subtitle_lines(
            str(item.get("text", "")).strip(),
            max_chars_per_line=req.max_chars_per_line,
            max_lines=req.max_lines_per_segment,
        )
        if not text:
            continue
        start_t = max(0.0, float(item.get("start_time", 0.0)))
        end_t = max(start_t + 0.25, float(item.get("end_time", start_t + 1.5)))

        seg = SubtitleSegment(
            id=str(uuid.uuid4()),
            project_id=req.project_id,
            index=idx,
            start_time=round(start_t, 3),
            end_time=round(end_t, 3),
            text=text,
            confidence=0.95,
            source="stt",
            words=[],
        )
        await db.insert_segment(seg)
        segments.append(seg)
        pct = min(98.0, 20.0 + (end_t / max(1.0, duration_sec)) * 78.0)
        await emit(
            WebSocketEvent(
                event="segment_created",
                project_id=req.project_id,
                job_id=job_id,
                progress=round(pct, 1),
                segment=seg,
            )
        )

    return segments


async def run_stt_pipeline(
    req: STTJobRequest,
    job_id: str,
    emit: EventCallback,
    cancel_event: asyncio.Event | None = None,
) -> list[SubtitleSegment]:
    """
    Execute the complete Speech-to-Text extraction pipeline for a project:
    1. Validate project and inspect video metadata.
    2. Extract and vocal-normalize 16kHz mono WAV audio.
    3. Transcribe with Faster-Whisper (or Gemini Cloud STT) using Silero VAD & Word Timestamps.
    4. Segment words into broadcast-ready subtitles, auto-saving each segment to SQLite
       and streaming events over WebSocket in real time.
    """
    cancel_ev = cancel_event or asyncio.Event()
    start_wall = time.perf_counter()

    project = await db.get_project(req.project_id)
    if not project:
        raise ValueError(f"Project not found: {req.project_id}")

    video_path = resolve_media_path(project.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    await db.update_project_status(
        req.project_id, status=JobStatus.RUNNING, progress=1.0, mode=ExtractionMode.STT
    )
    await db.clear_project_segments(req.project_id)

    await emit(
        WebSocketEvent(
            event="job_started",
            project_id=req.project_id,
            job_id=job_id,
            progress=1.0,
            message=f"Bắt đầu trích xuất âm thanh từ {video_path.name}...",
        )
    )

    try:
        meta = project.video_metadata or await asyncio.to_thread(probe_video, video_path)
        if not meta.has_audio:
            raise RuntimeError("Video không có luồng âm thanh (Audio Stream) để nhận diện giọng nói.")

        duration_sec = max(1.0, meta.duration_sec)

        # Step 1: Extract & normalize 16kHz Mono WAV
        await emit(
            WebSocketEvent(
                event="job_progress",
                project_id=req.project_id,
                job_id=job_id,
                progress=5.0,
                message="Đang tách và chuẩn hóa âm thanh 16kHz Mono (FFmpeg Vocal Filter)...",
            )
        )
        wav_path = await asyncio.to_thread(
            extract_audio_for_stt,
            video_path,
            None,
            req.normalize_audio,
            False,
        )

        if cancel_ev.is_set():
            await db.update_project_status(req.project_id, status=JobStatus.CANCELLED)
            await emit(
                WebSocketEvent(
                    event="job_cancelled",
                    project_id=req.project_id,
                    job_id=job_id,
                    message="Đã hủy tác vụ Speech-to-Text.",
                )
            )
            return []

        # Step 2: Check if Cloud STT is requested
        if req.use_cloud_api:
            segments = await _run_cloud_gemini_stt(
                wav_path, duration_sec, req, job_id, emit, cancel_ev
            )
            elapsed = round(time.perf_counter() - start_wall, 2)
            await db.update_project_status(req.project_id, status=JobStatus.COMPLETED, progress=100.0)
            await emit(
                WebSocketEvent(
                    event="job_completed",
                    project_id=req.project_id,
                    job_id=job_id,
                    progress=100.0,
                    message=f"Hoàn tất Cloud STT: {len(segments)} câu sub trong {elapsed}s.",
                    extra={"segment_count": len(segments), "elapsed_sec": elapsed},
                )
            )
            return segments

        # Step 3: Local Faster-Whisper Pipeline
        device, compute_type = resolve_stt_device_and_compute(req.device, req.compute_type)
        await emit(
            WebSocketEvent(
                event="job_progress",
                project_id=req.project_id,
                job_id=job_id,
                progress=10.0,
                message=f"Đang nạp model Faster-Whisper '{req.model_size}' trên {device.upper()} ({compute_type})...",
                extra={"device": device, "compute_type": compute_type, "model_size": req.model_size},
            )
        )

        loop = asyncio.get_running_loop()
        seg_queue: asyncio.Queue[SubtitleSegment | dict | None] = asyncio.Queue()
        detected_language_holder: list[str | None] = [None]

        segmenter = SmartWordSegmenter(
            max_chars_per_line=req.max_chars_per_line,
            max_lines_per_segment=req.max_lines_per_segment,
            max_pause_sec=req.max_pause_sec,
        )

        def _producer_worker() -> None:
            """Run Faster-Whisper generator in a worker thread and push segments to asyncio queue."""
            try:
                model, active_device, active_compute = model_cache.get_model(
                    req.model_size, device, compute_type
                )
                loop.call_soon_threadsafe(
                    seg_queue.put_nowait,
                    {
                        "type": "info",
                        "message": f"Model '{req.model_size}' sẵn sàng ({active_device.upper()}/{active_compute}). Đang nhận diện giọng nói...",
                    },
                )

                def _transcribe_with(m):
                    return m.transcribe(
                        str(wav_path),
                        language=req.language if req.language and req.language != "auto" else None,
                        beam_size=5,
                        vad_filter=req.vad_filter,
                        vad_parameters=dict(
                            min_silence_duration_ms=req.min_silence_duration_ms,
                            speech_pad_ms=req.speech_pad_ms,
                        ),
                        word_timestamps=req.word_timestamps,
                        condition_on_previous_text=False,
                        no_speech_threshold=req.no_speech_threshold,
                        compression_ratio_threshold=2.4,
                        initial_prompt=req.initial_prompt,
                    )

                try:
                    whisper_segments, info = _transcribe_with(model)
                    if info and hasattr(info, "language"):
                        det_lang = str(info.language)
                        det_prob = float(getattr(info, "language_probability", 1.0) or 1.0)
                        loop.call_soon_threadsafe(
                            seg_queue.put_nowait,
                            {
                                "type": "info",
                                "message": f"Phát hiện ngôn ngữ giọng nói: {det_lang.upper()} (Độ tin cậy: {int(det_prob * 100)}%)",
                                "detected_language": det_lang,
                            },
                        )
                    # Peek first segment or iterate with fallback if CUDA cublas fails at runtime
                    next_index = 1
                    for raw_seg in whisper_segments:
                        if cancel_ev.is_set():
                            break
                        next_index = _process_raw_whisper_segment(
                            raw_seg, req, segmenter, next_index, loop, seg_queue
                        )
                except Exception as runtime_exc:
                    if active_device == "cuda":
                        logger.warning(
                            "CUDA runtime error during transcription (%s). Retrying on CPU INT8...",
                            runtime_exc,
                        )
                        loop.call_soon_threadsafe(
                            seg_queue.put_nowait,
                            {
                                "type": "info",
                                "message": "GPU thiếu thư viện CUDA runtime, tự động chuyển sang CPU INT8...",
                            },
                        )
                        cpu_model, _, _ = model_cache.get_model(req.model_size, "cpu", "int8")
                        whisper_segments, info = _transcribe_with(cpu_model)
                        next_index = 1
                        for raw_seg in whisper_segments:
                            if cancel_ev.is_set():
                                break
                            next_index = _process_raw_whisper_segment(
                                raw_seg, req, segmenter, next_index, loop, seg_queue
                            )
                    else:
                        raise

                loop.call_soon_threadsafe(seg_queue.put_nowait, None)
            except Exception as err:
                loop.call_soon_threadsafe(
                    seg_queue.put_nowait, {"type": "error", "error": err}
                )
                loop.call_soon_threadsafe(seg_queue.put_nowait, None)

        worker_future = asyncio.to_thread(_producer_worker)
        worker_task = asyncio.create_task(worker_future)

        collected_segments: list[SubtitleSegment] = []

        while True:
            item = await seg_queue.get()
            if item is None:
                break

            if isinstance(item, dict):
                if item.get("type") == "error":
                    raise item["error"]
                if item.get("type") == "info":
                    if item.get("detected_language"):
                        detected_language_holder[0] = item["detected_language"]
                    await emit(
                        WebSocketEvent(
                            event="job_progress",
                            project_id=req.project_id,
                            job_id=job_id,
                            progress=12.0,
                            message=item["message"],
                            extra={"detected_language": item.get("detected_language")},
                        )
                    )
                continue

            if isinstance(item, SubtitleSegment):
                # Auto-save immediately to SQLite
                await db.insert_segment(item)
                collected_segments.append(item)

                pct = min(99.0, 12.0 + (item.end_time / duration_sec) * 87.0)
                await db.update_project_status(
                    req.project_id, status=JobStatus.RUNNING, progress=pct
                )
                await emit(
                    WebSocketEvent(
                        event="segment_created",
                        project_id=req.project_id,
                        job_id=job_id,
                        progress=round(pct, 1),
                        message=f"[{item.start_time:.1f}s -> {item.end_time:.1f}s] {item.text.replace(chr(10), ' ')}",
                        segment=item,
                    )
                )

        await worker_task

        if cancel_ev.is_set():
            await db.update_project_status(req.project_id, status=JobStatus.CANCELLED)
            await emit(
                WebSocketEvent(
                    event="job_cancelled",
                    project_id=req.project_id,
                    job_id=job_id,
                    message="Đã hủy tác vụ Speech-to-Text.",
                )
            )
            return collected_segments

        elapsed = round(time.perf_counter() - start_wall, 2)
        await db.update_project_status(
            req.project_id, status=JobStatus.COMPLETED, progress=100.0
        )

        # Auto-save export SRT file and record in extraction history
        export_srt_path = None
        try:
            srt_content = format_subtitles(collected_segments, ExportFormat.SRT)
            safe_name = re.sub(r'[\\/*?:"<>|]', '_', project.name)
            srt_filename = f"{safe_name}_{req.project_id[:8]}.srt"
            srt_file = settings.exports_dir / srt_filename
            srt_file.write_text(srt_content, encoding="utf-8")
            export_srt_path = str(srt_file)
        except Exception as export_err:
            logger.warning("Could not auto-export history SRT: %s", export_err)

        det_lang = detected_language_holder[0] or (req.language if req.language != "auto" else "vi")
        try:
            await db.add_history_record(
                project_id=req.project_id,
                mode="stt",
                language=req.language,
                detected_language=det_lang,
                segment_count=len(collected_segments),
                export_srt_path=export_srt_path,
                duration_sec=duration_sec,
                elapsed_sec=elapsed,
            )
        except Exception as hist_err:
            logger.warning("Could not add extraction history record: %s", hist_err)

        await emit(
            WebSocketEvent(
                event="job_completed",
                project_id=req.project_id,
                job_id=job_id,
                progress=100.0,
                message=f"Hoàn tất nhận diện giọng nói: {len(collected_segments)} câu sub trong {elapsed}s.",
                extra={
                    "segment_count": len(collected_segments),
                    "elapsed_sec": elapsed,
                    "rtf": round(elapsed / duration_sec, 3),
                    "detected_language": det_lang,
                },
            )
        )
        return collected_segments

    except Exception as exc:
        logger.exception("STT Pipeline failed for project %s", req.project_id)
        await db.update_project_status(req.project_id, status=JobStatus.FAILED)
        await emit(
            WebSocketEvent(
                event="job_failed",
                project_id=req.project_id,
                job_id=job_id,
                message=f"Lỗi Speech-to-Text: {exc}",
            )
        )
        raise


def _process_raw_whisper_segment(
    raw_seg,
    req: STTJobRequest,
    segmenter: SmartWordSegmenter,
    next_index: int,
    loop: asyncio.AbstractEventLoop,
    seg_queue: asyncio.Queue,
) -> int:
    """Convert a Faster-Whisper segment into 1 or more smart SubtitleSegments and enqueue them."""
    # Check no-speech probability guard
    no_speech_prob = getattr(raw_seg, "no_speech_prob", 0.0) or 0.0
    if no_speech_prob > 0.85 and (not raw_seg.text or len(raw_seg.text.strip()) < 5):
        return next_index

    raw_words = getattr(raw_seg, "words", None)
    if req.word_timestamps and req.smart_segmentation and raw_words:
        word_objs = [
            WordTimestamp(
                word=w.word,
                start=round(float(w.start), 3),
                end=round(float(w.end), 3),
                probability=round(float(getattr(w, "probability", 1.0)), 4),
            )
            for w in raw_words
            if w.word and w.word.strip()
        ]
        sub_segments = segmenter.segment_words(
            word_objs, project_id=req.project_id, start_index=next_index
        )
        for seg in sub_segments:
            loop.call_soon_threadsafe(seg_queue.put_nowait, seg)
            next_index += 1
        return next_index

    # Fallback when word_timestamps is disabled or empty
    text = (raw_seg.text or "").strip()
    if not text:
        return next_index

    avg_logprob = getattr(raw_seg, "avg_logprob", -0.2) or -0.2
    confidence = max(0.05, min(1.0, float(pow(2.71828, avg_logprob))))
    if is_likely_hallucination(text, confidence):
        return next_index

    formatted_text = wrap_subtitle_lines(
        text,
        max_chars_per_line=req.max_chars_per_line,
        max_lines=req.max_lines_per_segment,
    )
    seg = SubtitleSegment(
        id=str(uuid.uuid4()),
        project_id=req.project_id,
        index=next_index,
        start_time=round(float(raw_seg.start), 3),
        end_time=round(max(float(raw_seg.start) + 0.2, float(raw_seg.end)), 3),
        text=formatted_text,
        confidence=round(confidence, 4),
        source="stt",
        words=[],
    )
    loop.call_soon_threadsafe(seg_queue.put_nowait, seg)
    return next_index + 1
