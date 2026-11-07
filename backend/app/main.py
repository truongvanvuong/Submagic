"""FastAPI application entrypoint for Sub Magic."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import ensure_directories, settings
from app.database import db
from app.routers import export, history_api, jobs, models_api, projects


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    await db.init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API cho ứng dụng trích xuất phụ đề Sub Magic (Speech-to-Text & Hardsub OCR)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for cropped OCR frame images and exported files
app.mount("/static/crops", StaticFiles(directory=str(settings.crops_dir)), name="crops")
app.mount("/static/exports", StaticFiles(directory=str(settings.exports_dir)), name="exports")

app.include_router(models_api.router)
app.include_router(models_api.models_alias_router)
app.include_router(projects.router)
app.include_router(jobs.router)
app.include_router(export.router)
app.include_router(history_api.router)


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "ok", "version": settings.app_version}


# Serve pre-built frontend SPA if available for ultra-fast instant loading without Node/Vite
frontend_dist = settings.projects_dir.parent.parent.parent / "frontend" / "dist"
if not frontend_dist.exists():
    from pathlib import Path
    frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

