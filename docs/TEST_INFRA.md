# Sub Magic E2E Test Infrastructure

## Overview
This document specifies the End-to-End (E2E) testing architecture and infrastructure for **Sub Magic**, covering full export workflows, video media streaming (HTTP 200/206 Range), subtitle segment split/merge lifecycles, and Unicode/Vietnamese diacritic fidelity.

Testing is executed in an **opaque-box integration style** using FastAPI's `TestClient` (Starlette/HTTPX) against the actual SQLite database layer (`DatabaseManager`), subtitle formatter engine, and video file streaming endpoints.

---

## 4-Tier Testing Methodology

The test suite is structured into four distinct coverage tiers:

### Tier 1: Feature Coverage (≥5 test cases per feature)
Verifies primary happy-path behaviors and contractual guarantees for every major system capability:
1. **Subtitle Export Engine**:
   - TC1.1: Export SRT in single original layout (`none`).
   - TC1.2: Export VTT with `WEBVTT` header and period timecodes.
   - TC1.3: Export TXT plain text without timestamps.
   - TC1.4: Export TXT with inline timecodes (`include_timestamps_in_txt=True`).
   - TC1.5: Export Bilingual SRT (`original_top` vs `translated_top` vs `translated_only`).
   - TC1.6: GET `/api/projects/{id}/export` endpoint with standard query parameters and headers.
   - TC1.7: POST `/api/projects/{id}/export` endpoint with JSON payload and custom destination path.
2. **Video Streaming & Media Delivery**:
   - TC1.8: GET `/api/projects/{id}/video` returns HTTP 200 with `video/mp4` and full `Content-Length` when no Range is requested.
   - TC1.9: GET `/api/projects/{id}/video` with `Range: bytes=0-1023` returns HTTP 206 Partial Content, exact 1024 bytes, and `Content-Range: bytes 0-1023/{total}`.
   - TC1.10: GET `/api/projects/{id}/video` with suffix range `Range: bytes=-500` returns the trailing 500 bytes with HTTP 206.
   - TC1.11: GET `/api/projects/{id}/video` with open-ended range `Range: bytes=1000-` returns all bytes from offset 1000 to EOF.
   - TC1.12: GET `/api/projects/{id}/video` returns `Accept-Ranges: bytes` header for browser seeking support.
3. **Subtitle Segment Split & Merge**:
   - TC1.13: POST `/api/projects/{id}/segments/{seg_id}/split` divides a segment into two sequential segments with proper time intervals.
   - TC1.14: Segment split persists to SQLite and subsequent `GET /api/projects/{id}` contains both new segments.
   - TC1.15: POST `/api/projects/{id}/segments/merge` combines two adjacent segments, joining their text and extending bounds.
   - TC1.16: Segment merge persists to SQLite and deletes the merged-away segment.
   - TC1.17: Indices of all project segments remain strictly contiguous (1..N) after split and merge operations.
4. **Project Lifecycle & CRUD**:
   - TC1.18: Project creation with video probe metadata.
   - TC1.19: Manual segment addition via POST `/api/projects/{id}/segments`.
   - TC1.20: Segment text and translation update via PATCH `/api/projects/{id}/segments/{seg_id}`.
   - TC1.21: Segment deletion via DELETE `/api/projects/{id}/segments/{seg_id}` with re-indexing.
   - TC1.22: Project deletion cascades cleanly.

---

### Tier 2: Boundary & Corner Cases (≥5 test cases per feature)
Verifies resilience against extreme values, empty inputs, non-standard states, and hostile data:
1. **Subtitle Formatter Boundaries**:
   - TC2.1: Exporting an empty project (zero segments) returns valid empty files (empty string or bare `WEBVTT\n`).
   - TC2.2: Sub-millisecond timestamps (<0.001s) clamp gracefully to `00:00:00,000` / `00:00:00.000`.
   - TC2.3: Zero-duration or inverted segments (`end_time <= start_time`) clamp automatically to minimum duration (>=0.1s).
   - TC2.4: High-hour timestamps (>100 hours) format without overflow or negative numbers.
   - TC2.5: Missing translated text in bilingual export modes gracefully falls back to original text without trailing empty newlines.
2. **Video Streaming Boundaries**:
   - TC2.6: Range requests beyond file size (e.g., `bytes=999999999-`) return HTTP 416 Range Not Satisfiable.
   - TC2.7: Single-byte range request `Range: bytes=0-0` returns exactly 1 byte with HTTP 206.
   - TC2.8: Last-byte range request `Range: bytes={total-1}-{total-1}` returns exactly 1 byte.
   - TC2.9: Non-existent video file on disk returns HTTP 404 cleanly.
   - TC2.10: Zero-byte video file handles range headers without division-by-zero crashes.
3. **Segment Split & Merge Boundaries**:
   - TC2.11: Split at exact `start_time` or `end_time` rejects with HTTP 400 or clamps safely.
   - TC2.12: Split timestamp outside segment interval rejects with HTTP 400.
   - TC2.13: Merging non-existent segment IDs returns HTTP 400 / 404.
   - TC2.14: Merging identical segment ID with itself rejects cleanly.
   - TC2.15: Splitting a segment with empty text produces valid split segments with empty/default text.
4. **Unicode & Vietnamese Diacritics Resilience**:
   - TC2.16: Complex tone marks (ă, â, đ, ê, ô, ơ, ư, hỏi, ngã, nặng, sắc, huyền) preserve UTF-8 byte sequences identically across export formats.
   - TC2.17: Mixed scripts (Vietnamese + Japanese Kanji + Chinese Hanzi + Emoji) survive roundtrip export.
   - TC2.18: Multiline subtitle bodies with mixed CRLF/LF preserve line formatting without blank lines.
   - TC2.19: HTML-like tags (`<b>`, `<i>`, `<font>`) in subtitle text are preserved verbatim in SRT and VTT.
   - TC2.20: Content-Disposition headers with non-ASCII project names use RFC 5987 `filename*=UTF-8''...` encoding.

---

### Tier 3: Cross-Feature Combinations (Pairwise & Multi-Step Interactions)
Verifies that features operate coherently when chained:
1. **Split -> Edit -> Bilingual Export Flow**:
   - Create project -> Add segment with Vietnamese text -> Split at midpoint -> Update second segment with English translation -> Export as Bilingual SRT (`original_top`) -> Verify time continuity and bilingual pairing across both segments.
2. **Merge -> Re-index -> Streaming Sync Flow**:
   - Create project with sample video -> Add 3 segments -> Merge segments 1 and 2 -> Query video stream at merged start time -> Export VTT -> Verify webvtt cue 1 spans the entire merged interval.
3. **ROI Configuration -> Manual Segment -> Export Flow**:
   - Update project ROI with custom exclusion masks -> Add manual segments -> Export to TXT with timestamps -> Verify ROI update does not invalidate segments.
4. **Multi-Split Cascade Flow**:
   - Split segment 1 into 1A and 1B -> Split 1B into 1B1 and 1B2 -> Verify indices remain sequential 1, 2, 3 -> Verify export produces exactly 3 sequentially numbered cues.
5. **Concurrent Client Export Flow**:
   - Multiple parallel requests for SRT, VTT, and TXT on the same project return consistent byte outputs.

---

### Tier 4: Real-World Application Scenarios
Simulates realistic end-user production workflows:
1. **Scenario 1: Full Video Subtitling & Post-Editing Lifecycle**:
   - Upload/register an MP4 video file.
   - Simulate STT/OCR output: insert 5 realistic Vietnamese subtitle segments with speech pauses.
   - User reviews video via Range streaming endpoint (`Range: bytes=0-4096`).
   - User spots a long segment (6.5s) and splits it at 3.2s with appropriate text partition.
   - User translates segments into English (`translated_text`).
   - User exports final subtitles as:
     - `.srt` bilingual (`original_top`) for VLC / Media Player.
     - `.vtt` bilingual (`translated_top`) for web HTML5 `<video>` `<track>`.
     - `.txt` single (`translated_only`) for summary script.
   - Validate structure, timestamps, cues, and UTF-8 encoding of all three generated files.
2. **Scenario 2: HTML5 Video Seeking and Scrubber Playback**:
   - Simulate browser media player issuing sequential range requests:
     - Request initial metadata header (`bytes=0-1023`).
     - Seek to 50% (`bytes={half}-{half+8191}`).
     - Stream continuously to EOF.
   - Validate 206 status codes, `Content-Range`, `Content-Length`, and sequential byte alignment.
3. **Scenario 3: Corrupted Input Recovery**:
   - Project created with unprobed video file -> Request video -> Request export with empty segments -> Validate non-crashing behavior and structured HTTP error responses.

---

## Test Execution & Verification

### Test Harness
- Framework: `pytest` + `pytest-asyncio`
- Client: `fastapi.testclient.TestClient`
- Database: In-memory or temporary SQLite database initialized per test module.
- Video fixtures: Synthetic MP4 test video generated dynamically via OpenCV or FFmpeg lavfi.

### Runner Commands
```powershell
# Run the complete E2E integration suite
backend\.venv\Scripts\python.exe -m pytest backend/tests/test_e2e_integration.py -v

# Run with full traceback and short test report
backend\.venv\Scripts\python.exe -m pytest backend/tests/test_e2e_integration.py --tb=short -q

# Run all backend tests (Phase 1, 2, 3, 5, and E2E)
backend\.venv\Scripts\python.exe -m pytest backend/tests/ -v
```

### Verification Criteria
- 100% of E2E tests pass (`0 failures`, `0 errors`).
- Zero hardcoded fake responses: all tests assert against actual HTTP response bodies and SQLite database rows.
- Complete isolation: temporary files and databases are cleaned up via `tmp_path`.
