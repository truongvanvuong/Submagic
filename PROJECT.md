# Project: Submagic Hardsub OCR Accuracy & Timing Synchronization Optimization

## Architecture
Submagic is a FastAPI + PyAV/FFmpeg + RapidOCR/VietOCR hardsub extraction and translation platform running on CPU.
- **Video Ingestion & Stream Processing (`app/services/ffmpeg_service.py`)**:
  - Probes video metadata (`r_frame_rate`, `avg_frame_rate`, stream `start_time`, `fps`).
  - Demuxes presentation timestamps (PTS) via PyAV packet scanning.
  - Pipes decoded ROI frames via FFmpeg `-fps_mode passthrough` (`-vsync 0`) in BGR24.
- **OCR Hardsub Engine (`app/services/ocr_engines.py`, `app/services/ocr_pipeline.py`)**:
  - Multi-tier visual activity filter: Otsu/percentile adaptive thresholding on ROI luminance + relaxed stroke density bounds.
  - RapidOCR PP-OCRv4 parameterization (`text_score`, `box_thresh`, `thresh`, `unclip_ratio`).
  - Mask coverage horizontal overflow verifier & remediation trigger.
  - Temporal boundary binary search refined with frame-accurate PTS lookup table.
  - VietOCR transparent fallback detection & proactive warnings.
- **Deduplication & Text Normalization (`app/services/ocr_pipeline.py`, `app/schemas.py`, `app/database.py`)**:
  - CJK character-level comparison in `FuzzySubtitleDeduplicator` + mask visual difference guard.
  - OpenCC `t2s` normalization and dual-version text storage (`original_text`, `normalized_text`).
  - Script distribution counter and mixed-script low-confidence flagging.
  - Vietnamese full-resolution unbinarized crop preservation.
- **Translation Caching & Cloud AI Fallback (`app/database.py`, `app/services/llm_service.py`, `app/routers/jobs.py`)**:
  - 5-tuple composite cache key `(source_text, target_lang, context_type, model_name, prompt_version)`.
  - Cache clearing REST API endpoint and UI action.
  - Opt-in Gemini Vision OCR fallback capped by line count, gated by pre-activation cost estimation, and routed exclusively for low-confidence spans.
- **Evaluation & Measurement (`scripts/eval_srt.py`)**:
  - Precision, Recall, CER (standard and diacritic-stripped), start/end time offsets and drift trends.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Eval Tooling & Baseline Record | Add `--fold-script t2s`, `--vietnamese-diacritics` and UTF-8 encoding fix to `eval_srt.py`; record baseline metrics table | M0 | R1, R6 |
| 2 | Diagnostic Logging | Detailed drop reason logging (activity, text_score, min_confidence, merge), frame counts, OCR count, processing time | M1 | R1 |
| 3 | RapidOCR Parameterization | Configurable `text_score`, `box_thresh`, `thresh`, `unclip_ratio` in `OCRJobRequest` and `ocr_engines.py` | M1 | R1 |
| 4 | ROI Multi-Axis Padding | Add `pad_left_ratio`, `pad_right_ratio`, `pad_bottom_ratio` in `crop_and_mask_roi` and `VideoROIStream` (protect Vietnamese dấu nặng) | M1 | R1 |
| 5 | VietOCR Fallback Transparency | Proactive warning log and WebSocket alert when `engine=auto` & `language=vi` without `vietocr` | M1 | R1 |
| 6 | Target Case 誰 Recovery | Fix detection/recognition clipping for "誰能想到東方不敗也敗在別人手下了" | M1 | R1 |
| 7 | Video Probe Diagnostics | Parse & log `video_fps`, `r_frame_rate`, `avg_frame_rate`, stream `start_time`, expected vs actual frame count | M2 | R2 |
| 8 | FFmpeg Passthrough Mode | Run FFmpeg with `-fps_mode passthrough` (`-vsync 0`) to prevent dropped/duplicated frames | M2 | R2 |
| 9 | PyAV Packet PTS Demuxer | Extract monotonic frame presentation timestamp table from video stream packets | M2 | R2 |
| 10 | PTS Table Lookup for Timing | Replace `frame_idx / fps` with PTS table indexing for start_time, end_time, and binary search; handle non-zero start_time | M2 | R2 |
| 11 | Mask Coverage Verification | Compare horizontal extent of text mask vs union of OCR bounding boxes; detect left/right overflow | M3 | R3 |
| 12 | Overflow Remediation Trigger | Retry OCR with relaxed parameters or edge crop when mask overflows bounding box | M3 | R3 |
| 13 | Adaptive ROI Thresholding | Otsu and 85th percentile adaptive thresholding on ROI luminance replacing fixed 185 | M3 | R3 |
| 14 | Relaxed Stroke Density Range | Expand text activity density from `[0.001, 0.65]` to `[0.001, 0.85]` on verified stroke edges | M3 | R3 |
| 15 | CJK Character-Level Deduplication | Enforce character-by-character comparison in `FuzzySubtitleDeduplicator` for CJK length >= 7 differing by 1 char | M3 | R3 |
| 16 | Mask Visual Difference Guard | Block segment merging if visual mask difference between frames exceeds threshold | M3 | R3 |
| 17 | OpenCC `t2s` Normalization | Convert Traditional to Simplified Chinese for deduplication, scoring, and cache key generation | M4 | R4 |
| 18 | Dual-Version Subtitle Storage | Store `original_text` and `normalized_text` in database and schemas; add export format selector | M4 | R4 |
| 19 | Script Distribution Counter | Count distinct Traditional and Simplified glyphs per line | M4 | R4 |
| 20 | Mixed-Script Low-Confidence Flag | Automatically flag segments as low confidence when mixing Traditional and Simplified characters | M4 | R4 |
| 21 | Vietnamese Full-Resolution Crop | Preserve unbinarized full-resolution crop for Vietnamese OCR to protect dấu nặng | M4 | R4 |
| 22 | Dual CER Evaluator | Support standard CER and diacritic-stripped CER in `eval_srt.py` | M4 | R4 |
| 23 | Translation Cache 5-Tuple Key | Extend cache composite key to `(source_text, target_lang, context_type, model_name, prompt_version)` with `t2s` normalized text | M5 | R5 |
| 24 | Translation Cache Clear Endpoint | REST API endpoint and UI action to purge translation cache | M5 | R5 |
| 25 | Selective Gemini OCR Fallback | Route low-confidence spans to Gemini Vision OCR (default off, capped by max_lines) | M5 | R5 |
| 26 | Gemini OCR Cost Estimator | Pre-compute token and USD cost before user enables Gemini OCR | M5 | R5 |
| 27 | Final E2E Verification & Comparison | Run 100% pytest suite, benchmark final metrics vs baseline on sample video, deliver checklist | M6 | R6 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Baseline & Eval Tooling | UTF-8 console, `--fold-script t2s`, baseline metric recording | None | DONE |
| M1 | Diagnostic Logging & RapidOCR Config | Diagnostic logs, RapidOCR parameters, ROI padding, VietOCR warning, generic mask recovery | M0 | DONE |
| M2 | Precise PTS Timestamp Sync | Probed rates, PyAV PTS demuxer, FFmpeg passthrough, PTS lookup in boundary search | M1 | DONE (316/316 tests pass, 5/5 Gate unanimous PASS) |
| M3 | Mask Coverage & CJK Dedup | Mask overflow check & retry, adaptive threshold, CJK character-level dedup | M2 | DONE (360/360 tests pass) |
| M4 | OpenCC Normalization & Dual Storage | OpenCC `t2s`, dual-version storage, script counter, Vietnamese full-res & dual CER | M3 | DONE (369/369 tests pass) |
| M5 | Cache 5-Tuple Key & Gemini OCR | Extended cache key, cache clear endpoint, selective Gemini OCR fallback & cost estimation | M4 | DONE (4/4 tests pass) |
| M6 | Final Verification & Benchmark | Full pytest suite pass, final before/after comparison table, self-check guide | M5 | DONE (141/141 tests pass, 100% verified) |

## Interface Contracts
### `app/services/ffmpeg_service.py` ↔ `app/services/ocr_pipeline.py`
- `extract_video_pts_table(video_path: str | Path, stream_index: int = 0) -> list[float]`
  - Returns strictly monotonic presentation timestamps in seconds normalized to playback zero.
  - Fallback: empty list (caller falls back to `frame_idx / fps`).
- `VideoROIStream(..., pad_left_ratio: float = 0.0, pad_right_ratio: float = 0.0, pad_bottom_ratio: float = 0.0)`
  - Adds symmetric/asymmetric horizontal and vertical padding clamped within safe ratios.
  - Yields tuples `(frame_idx, pts_timestamp, roi_bgr)`.

### `app/services/ocr_engines.py` ↔ `app/schemas.py` & `app/services/ocr_pipeline.py`
- `OCRJobRequest`:
  - `rapidocr_text_score: float = 0.5`
  - `rapidocr_box_thresh: float = 0.5`
  - `rapidocr_thresh: float = 0.3`
  - `rapidocr_unclip_ratio: float = 1.6`
  - `pad_left_ratio: float = 0.0`
  - `pad_right_ratio: float = 0.0`
  - `pad_bottom_ratio: float = 0.0`
- `verify_mask_coverage(roi_mask: np.ndarray, ocr_lines: list[OCRTextLine], min_overflow_ratio: float = 0.85) -> tuple[bool, str | None, int]`

### `app/schemas.py` & `app/database.py` ↔ Subtitle Output
- `SubtitleSegment`:
  - `original_text: str | None = None`
  - `normalized_text: str | None = None`
  - `confidence: float = 1.0`
  - `is_low_confidence: bool = False`

### `app/database.py` ↔ `app/services/llm_service.py`
- `translation_cache`:
  - Primary key: `(source_text, target_lang, context_type, model_name, prompt_version)`
  - `get_cached_translation(...)`, `set_cached_translation(...)`, `clear_translation_cache(...)`

## Code Layout
- `backend/app/schemas.py`: Data models, OCR request parameters, export options.
- `backend/app/database.py`: SQLite schema, migrations for `subtitle_segments` and `translation_cache`.
- `backend/app/services/ffmpeg_service.py`: Video probing, PyAV PTS demuxer, FFmpeg ROI stream.
- `backend/app/services/ocr_engines.py`: RapidOCR wrapper, parameter forwarding, mask coverage check, VietOCR warning.
- `backend/app/services/ocr_pipeline.py`: OCR pipeline orchestration, adaptive thresholding, PTS boundary refinement, deduplication.
- `backend/app/routers/jobs.py`: OCR job triggers, cache clear endpoint, Gemini cost estimator.
- `backend/app/routers/export.py`: Multi-format subtitle export with script variant options.
- `scripts/eval_srt.py`: SRT evaluation CLI supporting `--fold-script` and `--vietnamese-diacritics`.
- `backend/tests/`: Pytest unit and integration test suite.
