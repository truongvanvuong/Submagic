"""Pydantic schemas for Sub Magic REST API and WebSocket events."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class ExtractionMode(str, Enum):
    STT = "stt"
    OCR = "ocr"
    HYBRID = "hybrid"


class JobStatus(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"


class BilingualLayout(str, Enum):
    NONE = "none"
    ORIGINAL_TOP = "original_top"
    TRANSLATED_TOP = "translated_top"
    TRANSLATED_ONLY = "translated_only"


class HardwareInfo(BaseModel):
    os_name: str
    cpu_name: str
    cpu_cores: int
    ram_total_gb: float
    cuda_available: bool
    cuda_device_count: int = 0
    gpu_names: list[str] = Field(default_factory=list)
    onnx_providers: list[str] = Field(default_factory=list)
    recommended_stt_device: Literal["cuda", "cpu"] = "cpu"
    recommended_stt_compute_type: str = "int8"
    recommended_ocr_provider: str = "CPUExecutionProvider"
    ffmpeg_available: bool = False
    ffmpeg_path: str | None = None


class VideoMetadata(BaseModel):
    file_path: str
    file_name: str
    file_size_bytes: int
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    video_codec: str | None = None
    has_audio: bool = True
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None


class NormalizedBox(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0 relative to video width/height)."""
    x: float = Field(ge=0.0, le=1.0, description="Left offset (0.0 - 1.0)")
    y: float = Field(ge=0.0, le=1.0, description="Top offset (0.0 - 1.0)")
    w: float = Field(gt=0.0, le=1.0, description="Width (0.0 - 1.0)")
    h: float = Field(gt=0.0, le=1.0, description="Height (0.0 - 1.0)")
    label: str | None = None

    def to_pixel_rect(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Convert normalized box to (x1, y1, x2, y2) clamped to frame dimensions."""
        x1 = max(0, min(frame_width - 1, int(round(self.x * frame_width))))
        y1 = max(0, min(frame_height - 1, int(round(self.y * frame_height))))
        x2 = max(x1 + 1, min(frame_width, int(round((self.x + self.w) * frame_width))))
        y2 = max(y1 + 1, min(frame_height, int(round((self.y + self.h) * frame_height))))
        return x1, y1, x2, y2


class ROIConfig(BaseModel):
    """Region of Interest configuration for Hardsub OCR."""
    main_roi: NormalizedBox = Field(
        default_factory=lambda: NormalizedBox(x=0.08, y=0.78, w=0.84, h=0.18, label="Main Subtitle ROI")
    )
    exclusion_masks: list[NormalizedBox] = Field(
        default_factory=list,
        description="List of exclusion rectangles to mask out logos/watermarks inside main_roi",
    )


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class SubtitleSegment(BaseModel):
    id: str | None = None
    project_id: str
    index: int
    start_time: float = Field(ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(ge=0.0, description="End timestamp in seconds")
    text: str
    translated_text: str | None = None
    confidence: float = 1.0
    source: Literal["stt", "ocr", "manual"] = "stt"
    crop_image_url: str | None = None
    words: list[WordTimestamp] = Field(default_factory=list)


class SubtitleSegmentUpdate(BaseModel):
    start_time: float | None = None
    end_time: float | None = None
    text: str | None = None
    translated_text: str | None = None
    confidence: float | None = None
    crop_image_url: str | None = None



class STTJobRequest(BaseModel):
    project_id: str
    model_size: Literal[
        "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"
    ] = "small"
    language: str | None = Field(
        default=None,
        description="ISO language code (e.g., 'vi', 'en', 'zh', 'ja', 'ko') or None for auto-detect",
    )
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    normalize_audio: bool = True
    vad_filter: bool = True
    min_silence_duration_ms: int = 400
    speech_pad_ms: int = 200
    no_speech_threshold: float = 0.6
    word_timestamps: bool = True
    smart_segmentation: bool = True
    max_chars_per_line: int = 42
    max_lines_per_segment: int = 2
    max_pause_sec: float = 0.45
    initial_prompt: str | None = None
    use_cloud_api: bool = False
    gemini_api_key: str | None = None


class OCRJobRequest(BaseModel):
    project_id: str
    language: Literal["auto", "vi", "en", "ch", "ja", "ko", "multi"] = "auto"
    engine: Literal["auto", "rapidocr", "vietocr", "gemini"] = "auto"
    sample_fps: float = Field(default=3.0, ge=1.0, le=30.0)
    roi_config: ROIConfig = Field(default_factory=ROIConfig)
    luma_threshold: int = Field(default=185, ge=50, le=255)
    stability_frames: int = Field(default=1, ge=0, le=5)
    visual_diff_threshold: float = Field(default=0.14, ge=0.01, le=0.90)
    fuzzy_match_threshold: float = Field(default=85.0, ge=50.0, le=100.0)
    max_merge_gap_sec: float = Field(default=0.20, ge=0.0, le=2.0)
    save_crop_images: bool = True
    gemini_api_key: str | None = None


class LLMJobRequest(BaseModel):
    project_id: str
    fix_ocr_spelling: bool = True
    translate_to: str | None = Field(
        default=None,
        description="Target language for bilingual translation (e.g., 'vi', 'en', 'zh', 'ja', 'ko')",
    )
    batch_size: int = Field(default=60, ge=1, le=100)
    gemini_api_key: str | None = None
    model: str | None = Field(
        default=None,
        description="Optional Gemini model override (e.g., 'gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-2.5-flash')",
    )



class ProjectCreateRequest(BaseModel):
    name: str | None = None
    video_path: str
    mode: ExtractionMode = ExtractionMode.STT
    language: str | None = "vi"


class ProjectSummary(BaseModel):
    id: str
    name: str
    video_path: str
    video_metadata: VideoMetadata | None = None
    mode: ExtractionMode
    language: str | None = None
    roi_config: ROIConfig = Field(default_factory=ROIConfig)
    status: JobStatus = JobStatus.IDLE
    progress: float = 0.0
    segment_count: int = 0
    created_at: str
    updated_at: str


class ProjectDetail(ProjectSummary):
    segments: list[SubtitleSegment] = Field(default_factory=list)


class ExportRequest(BaseModel):
    format: ExportFormat = ExportFormat.SRT
    bilingual_layout: BilingualLayout = BilingualLayout.NONE
    include_timestamps_in_txt: bool = False
    save_to_disk_path: str | None = None


class WebSocketEvent(BaseModel):
    event: Literal[
        "job_started",
        "job_progress",
        "segment_created",
        "segment_updated",
        "job_completed",
        "job_failed",
        "job_cancelled",
    ]
    project_id: str
    job_id: str | None = None
    progress: float | None = None
    message: str | None = None
    segment: SubtitleSegment | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
