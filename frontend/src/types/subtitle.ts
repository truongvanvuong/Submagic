/**
 * Types and interfaces for Sub Magic Subtitle application.
 * Mirrors backend Pydantic models from backend/app/schemas.py.
 */

export type ExtractionMode = 'stt' | 'ocr' | 'hybrid';

// Ambient declaration for Vite's import.meta.env
declare global {
  interface ImportMeta {
    env: {
      VITE_API_BASE_URL?: string;
      VITE_WS_BASE_URL?: string;
      [key: string]: any;
    };
  }
}

export type JobStatus =
  | 'idle'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

/** Alias for JobStatus representing the status of a project. */
export type ProjectStatus = JobStatus;

export type ExportFormat = 'srt' | 'vtt' | 'txt';

export type BilingualLayout =
  | 'none'
  | 'original_top'
  | 'translated_top'
  | 'translated_only';

/**
 * On-video subtitle overlay visual & language settings.
 */
export interface SubtitleOverlaySettings {
  enabled: boolean;
  fontSize: 'small' | 'medium' | 'large';
  position: 'bottom' | 'center' | 'top';
  background: 'none' | 'translucent' | 'solid';
  languageMode: 'original' | 'translated' | 'bilingual';
}

export const DEFAULT_SUBTITLE_SETTINGS: SubtitleOverlaySettings = {
  enabled: true,
  fontSize: 'medium',
  position: 'bottom',
  background: 'translucent',
  languageMode: 'bilingual',
};

/**
 * Normalized bounding box coordinates (0.0 to 1.0 relative to video dimensions).
 */
export interface NormalizedBox {
  x: number; // 0.0 - 1.0 (Left offset relative to video width)
  y: number; // 0.0 - 1.0 (Top offset relative to video height)
  w: number; // 0.0 - 1.0 (Width)
  h: number; // 0.0 - 1.0 (Height)
  label?: string | null;
}

/**
 * Region of Interest configuration for Hardsub OCR.
 */
export interface ROIConfig {
  main_roi: NormalizedBox;
  exclusion_masks: NormalizedBox[];
}

/**
 * Word-level timestamp for STT transcription alignment.
 */
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
  probability?: number;
}

/**
 * Single subtitle segment entity.
 */
export interface SubtitleSegment {
  id: string;
  project_id: string;
  index: number;
  start_time: number;
  end_time: number;
  text: string;
  translated_text?: string | null;
  confidence: number;
  source: 'stt' | 'ocr' | 'manual';
  crop_image_url?: string | null;
  words?: WordTimestamp[];
}

/**
 * Partial update for a subtitle segment.
 */
export interface SubtitleSegmentUpdate {
  start_time?: number;
  end_time?: number;
  text?: string;
  translated_text?: string | null;
  confidence?: number;
  crop_image_url?: string | null;
}

/**
 * Request payload for splitting a segment.
 */
export interface SubtitleSegmentSplitRequest {
  split_time: number;
  first_text?: string;
  second_text?: string;
  first_translated?: string | null;
  second_translated?: string | null;
}

/**
 * Video technical metadata.
 */
export interface VideoMetadata {
  file_path: string;
  file_name: string;
  file_size_bytes: number;
  duration_sec: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  video_codec?: string | null;
  has_audio: boolean;
  audio_codec?: string | null;
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
}

/**
 * Hardware capability information detected by the backend.
 */
export interface HardwareInfo {
  os_name: string;
  cpu_name: string;
  cpu_cores: number;
  ram_total_gb: number;
  cuda_available: boolean;
  cuda_device_count: number;
  gpu_names: string[];
  onnx_providers: string[];
  recommended_stt_device: 'cuda' | 'cpu';
  recommended_stt_compute_type: string;
  recommended_ocr_provider: string;
  ffmpeg_available: boolean;
  ffmpeg_path?: string | null;
}

/**
 * Speech-to-Text configuration request.
 */
export interface STTJobRequest {
  project_id: string;
  model_size?:
    | 'tiny'
    | 'base'
    | 'small'
    | 'medium'
    | 'large-v3'
    | 'large-v3-turbo'
    | 'distil-large-v3';
  language?: string | null;
  device?: 'auto' | 'cuda' | 'cpu';
  compute_type?: string;
  normalize_audio?: boolean;
  vad_filter?: boolean;
  min_silence_duration_ms?: number;
  speech_pad_ms?: number;
  no_speech_threshold?: number;
  word_timestamps?: boolean;
  smart_segmentation?: boolean;
  max_chars_per_line?: number;
  max_lines_per_segment?: number;
  max_pause_sec?: number;
  initial_prompt?: string | null;
  use_cloud_api?: boolean;
  gemini_api_key?: string | null;
}

/**
 * Hardsub OCR configuration request.
 */
export interface OCRJobRequest {
  project_id: string;
  language?: 'vi' | 'en' | 'ch' | 'ja' | 'ko' | 'multi' | 'auto';
  engine?: 'auto' | 'rapidocr' | 'vietocr' | 'gemini';
  sample_fps?: number;
  roi_config?: ROIConfig;
  luma_threshold?: number;
  stability_frames?: number;
  visual_diff_threshold?: number;
  fuzzy_match_threshold?: number;
  max_merge_gap_sec?: number;
  save_crop_images?: boolean;
  gemini_api_key?: string | null;
}

/**
 * LLM Post-processing (spelling fix / bilingual translation) request.
 */
export interface LLMJobRequest {
  project_id: string;
  fix_ocr_spelling?: boolean;
  translate_to?: string | null;
  batch_size?: number;
  gemini_api_key?: string | null;
}

/**
 * Request payload for creating a new project.
 */
export interface ProjectCreateRequest {
  name?: string | null;
  video_path: string;
  mode?: ExtractionMode;
  language?: string | null;
}

/**
 * Summary of a project (without segments array).
 */
export interface ProjectSummary {
  id: string;
  name: string;
  video_path: string;
  video_metadata?: VideoMetadata | null;
  mode: ExtractionMode;
  language?: string | null;
  roi_config: ROIConfig;
  status: JobStatus;
  progress: number;
  segment_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * Full project details including segments list.
 */
export interface ProjectDetail extends ProjectSummary {
  segments: SubtitleSegment[];
}

/**
 * Subtitle export parameters.
 */
export interface ExportRequest {
  format: ExportFormat;
  bilingual_layout: BilingualLayout;
  include_timestamps_in_txt: boolean;
  save_to_disk_path?: string | null;
}

/**
 * WebSocket event stream schema.
 */
export interface WebSocketEvent {
  event:
    | 'job_started'
    | 'job_progress'
    | 'segment_created'
    | 'segment_updated'
    | 'job_completed'
    | 'job_failed'
    | 'job_cancelled';
  project_id: string;
  job_id?: string | null;
  progress?: number | null;
  message?: string | null;
  error?: string | null;
  segment?: SubtitleSegment | null;
  extra?: Record<string, any>;
}

/**
 * Whisper model metadata in the model catalog.
 */
export interface WhisperModelInfo {
  id: string;
  name: string;
  speed: string;
  accuracy: string;
  vram_gb: number;
  downloaded?: boolean;
}

/**
 * OCR Engine metadata in the model catalog.
 */
export interface OCREngineInfo {
  id: string;
  name: string;
  local: boolean;
}

/**
 * Response structure for GET /api/system/models.
 */
export interface ModelCatalogResponse {
  whisper_models: WhisperModelInfo[];
  ocr_engines: OCREngineInfo[];
}

/**
 * Response returned when launching a background job.
 */
export interface JobStatusResponse {
  job_id: string;
  project_id: string;
  status: JobStatus;
  message: string;
}

/**
 * Extraction history item.
 */
export interface ExtractionHistoryItem {
  id: string;
  project_id: string;
  project_name: string;
  video_path: string;
  mode: ExtractionMode;
  language?: string | null;
  detected_language?: string | null;
  segment_count: number;
  export_srt_path?: string | null;
  export_vtt_path?: string | null;
  duration_sec: number;
  elapsed_sec: number;
  status: string;
  created_at: string;
}
