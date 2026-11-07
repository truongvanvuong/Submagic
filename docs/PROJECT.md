# Project: Sub Magic (Phases 4 & 5)

## Architecture
Sub Magic is an AI-powered automated video subtitling application supporting both Speech-to-Text (STT) and Hardsub Video OCR, with bilingual translation, LLM subtitle polishing, interactive ROI/mask configuration, real-time progress streaming, interactive subtitle editing, and multi-format subtitle export.

### System Diagram
```
[ Frontend: React 18 + TypeScript + Vite + TailwindCSS ]
  ├── App.tsx (Main orchestrator & layout container)
  ├── ControlPanel.tsx (Mode, language, model selection)
  ├── VideoWorkspace.tsx (HTML5 <video> + Overlay Canvas for ROI / Exclusion Masks)
  ├── SubtitleEditor.tsx (Timestamp sync, OCR crop previews, merge/split, AI polish)
  ├── ProgressBar.tsx (Real-time progress percentage & log streaming)
  └── ExportModal.tsx (SRT, VTT, TXT single & bilingual export options)
        │ REST API (fetch/axios)              │ WebSocket (/api/jobs/ws/{project_id})
        ▼                                      ▼
[ Backend: FastAPI (Python 3.11) + SQLite (aiosqlite) ]
  ├── app/api/projects.py (CRUD, Range video streaming, Frame preview, Segments split/merge)
  ├── app/api/jobs.py (STT/OCR/LLM execution pipelines, WebSocket manager)
  ├── app/api/export.py (GET/POST subtitle export in SRT, VTT, TXT)
  ├── app/utils/subtitle_formatter.py (Formatters & timecode parsing)
  └── static mounts (/static/crops/{project_id}/..., /static/exports/...)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Subtitle Formatter Enhancement | Support SRT, VTT, TXT for single & bilingual (original_top, translated_top, translated_only), timecode parser (`parse_timestamp`), text wrapping | M1 | Survey |
| 2 | Export Endpoint Completion | Add `GET /api/projects/{id}/export` with query params, proper MIME types (`application/x-subrip`, `text/vtt`, `text/plain`), and Content-Disposition | M1 | Survey |
| 3 | Video Streaming Endpoint | Add `GET /api/projects/{id}/video` supporting HTTP 206 Partial Content Range streaming via `FileResponse` | M1 | Survey |
| 4 | Subtitle Segment Split API | Add `POST /api/projects/{id}/segments/{id}/split` and `db.split_segment` in database layer | M1 | Survey |
| 5 | Windows Startup Script | Create `start.bat` to launch backend & frontend concurrently and open default browser at `http://localhost:5173` | M1 | Survey |
| 6 | Frontend Project Bootstrap | Initialize React + TypeScript + Vite + TailwindCSS in `frontend/` with standard configs and clean light-mode theme | M2 | Survey |
| 7 | Subtitle Types Definition | Create `src/types/subtitle.ts` mirroring backend Pydantic schemas (`Project`, `SubtitleSegment`, `NormalizedBox`, `ROIConfig`, etc.) | M2 | Survey |
| 8 | REST & WebSocket API Service | Create `src/services/api.ts` with typed methods for all backend endpoints and WebSocket client for real-time progress & live segments | M2 | Survey |
| 9 | ControlPanel Component | Configuration for STT / OCR modes, language selection, AI model selection, execution triggers | M2 | Survey |
| 10 | ProgressBar Component | Real-time progress bar, status text, and live scrolling execution log stream via WebSocket | M2 | Survey |
| 11 | ExportModal Component | Dialog modal for subtitle export with format choices (.SRT, .VTT, .TXT), bilingual layouts, and download action | M2 | Survey |
| 12 | VideoWorkspace Component | HTML5 video player bound to `/api/projects/{id}/video` with overlay canvas: green ROI box, yellow exclusion masks, 16:9 & 9:16 presets, normalized coordinates | M3 | Survey |
| 13 | SubtitleEditor Component | Light-mode table/card list synchronized with video playback, active timestamp highlight, OCR crop image preview (`/static/crops/...`), Merge/Split actions, AI polish/translate buttons | M3 | Survey |
| 14 | App Integration & State | Main layout connecting ControlPanel, VideoWorkspace, SubtitleEditor, ProgressBar, ExportModal with global project state | M3 | Survey |
| 15 | E2E Testing Suite | Comprehensive test suite verifying backend export, video streaming, API contracts, and frontend build verification | M4 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Backend Export & Media Streaming Integration | Subtitle formatting (.SRT/.VTT/.TXT single & bilingual), GET/POST export endpoints, HTTP Range video streaming, segment split API, backend tests, and `start.bat` | none | DONE |
| M2 | Frontend Core & Service Layer | Bootstrap `frontend/`, `src/types/subtitle.ts`, `src/services/api.ts`, `ControlPanel.tsx`, `ProgressBar.tsx`, `ExportModal.tsx`, TypeScript verification | none | DONE |
| M3 | Interactive Workspace & Subtitle Editor | `VideoWorkspace.tsx` (canvas ROI/masks, presets), `SubtitleEditor.tsx` (sync, crop preview, merge/split), `App.tsx` integration, light-mode UI, build verification | M2 | DONE |
| M4 | Final Milestone: E2E Integration & Verification | Pass 100% E2E tests, end-to-end integration validation, adversarial coverage hardening | M1, M3 | DONE |

## Interface Contracts
### Frontend ↔ Backend REST API
- `GET /api/projects`: List all projects -> `ProjectSummary[]`
- `POST /api/projects`: Create project -> `ProjectDetail`
- `POST /api/projects/upload`: Upload video file -> `ProjectDetail`
- `GET /api/projects/{id}`: Get project details & segments -> `ProjectDetail`
- `GET /api/projects/{id}/video`: Stream video with HTTP 206 Range -> `video/mp4` (or original mime)
- `PUT /api/projects/{id}/roi`: Update ROI & Exclusion masks -> `ProjectDetail`
- `PATCH /api/projects/{id}/segments/{seg_id}`: Update segment text -> `SubtitleSegment`
- `POST /api/projects/{id}/segments/merge`: Merge multiple segments -> `SubtitleSegment[]`
- `POST /api/projects/{id}/segments/{seg_id}/split`: Split segment at timestamp -> `SubtitleSegment[]`
- `POST /api/jobs/stt`: Start STT job -> `JobStatusResponse`
- `POST /api/jobs/ocr`: Start OCR job -> `JobStatusResponse`
- `POST /api/jobs/llm`: Start AI polish/translation job -> `JobStatusResponse`
- `GET /api/projects/{id}/export?format={srt|vtt|txt}&bilingual_layout={...}`: Download subtitle file

### Frontend ↔ Backend WebSocket Protocol
- URL: `ws://localhost:8000/api/jobs/ws/{project_id}`
- Events:
  - `job_started`: `{ "event": "job_started", "data": { "job_id": string, "job_type": string } }`
  - `job_progress`: `{ "event": "job_progress", "data": { "job_id": string, "progress": number, "message": string } }`
  - `segment_created`: `{ "event": "segment_created", "data": { "segment": SubtitleSegment } }`
  - `segment_updated`: `{ "event": "segment_updated", "data": { "segment": SubtitleSegment } }`
  - `job_completed`: `{ "event": "job_completed", "data": { "job_id": string, "result": object } }`
  - `job_failed`: `{ "event": "job_failed", "data": { "job_id": string, "error": string } }`

## Code Layout
- Backend files:
  - `backend/app/utils/subtitle_formatter.py` (Owned by M1)
  - `backend/app/api/export.py` (Owned by M1)
  - `backend/app/api/projects.py` (Owned by M1)
  - `backend/app/database.py` (Owned by M1)
  - `backend/tests/test_phase5_export_and_media.py` (Owned by M1)
  - `start.bat` (Owned by M1)
- Frontend files:
  - `frontend/package.json`, `tsconfig.json`, `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `index.html` (Owned by M2)
  - `frontend/src/types/subtitle.ts` (Owned by M2)
  - `frontend/src/services/api.ts` (Owned by M2)
  - `frontend/src/components/ControlPanel.tsx` (Owned by M2)
  - `frontend/src/components/ProgressBar.tsx` (Owned by M2)
  - `frontend/src/components/ExportModal.tsx` (Owned by M2)
  - `frontend/src/components/VideoWorkspace.tsx` (Owned by M3)
  - `frontend/src/components/SubtitleEditor.tsx` (Owned by M3)
  - `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/src/index.css` (Owned by M3)
