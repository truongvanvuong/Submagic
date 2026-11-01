"""Hardware detection and AI Model management API routes."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import HardwareInfo
from app.services.hardware import detect_hardware

router = APIRouter(prefix="/api/system", tags=["system"])
models_alias_router = APIRouter(prefix="/api/models", tags=["models"])

WHISPER_MODELS_CATALOG = [
    {
        "id": "tiny",
        "name": "Whisper Tiny (~75 MB)",
        "speed": "Rất nhanh (Siêu nhẹ)",
        "accuracy": "Cơ bản",
        "vram_gb": 1.0,
    },
    {
        "id": "base",
        "name": "Whisper Base (~145 MB)",
        "speed": "Nhanh",
        "accuracy": "Khá",
        "vram_gb": 1.2,
    },
    {
        "id": "small",
        "name": "Whisper Small (~465 MB - Khuyên dùng CPU/GPU phổ thông)",
        "speed": "Cân bằng",
        "accuracy": "Tốt (Đa ngôn ngữ, Tiếng Việt/Anh/CJK)",
        "vram_gb": 2.0,
    },
    {
        "id": "medium",
        "name": "Whisper Medium (~1.5 GB)",
        "speed": "Trung bình",
        "accuracy": "Rất tốt",
        "vram_gb": 4.5,
    },
    {
        "id": "large-v3-turbo",
        "name": "Whisper Large-v3 Turbo (~1.6 GB - Khuyên dùng khi có GPU)",
        "speed": "Nhanh trên GPU",
        "accuracy": "Xuất sắc",
        "vram_gb": 5.0,
    },
    {
        "id": "large-v3",
        "name": "Whisper Large-v3 (~3.1 GB)",
        "speed": "Nặng",
        "accuracy": "Cao nhất",
        "vram_gb": 8.0,
    },
]


@router.get("/hardware", response_model=HardwareInfo)
@models_alias_router.get("/hardware", response_model=HardwareInfo)
async def get_hardware_info() -> HardwareInfo:
    """Return detected CPU, RAM, GPU CUDA, ONNX providers, and FFmpeg status."""
    return detect_hardware()


@router.get("/models")
@models_alias_router.get("")
@models_alias_router.get("/")
async def list_available_models() -> dict:
    """List supported STT and OCR models and check local cache presence."""
    whisper_dir = settings.models_dir / "whisper"
    cached_folders = (
        [p.name for p in whisper_dir.iterdir() if p.is_dir()]
        if whisper_dir.exists()
        else []
    )

    models = []
    for item in WHISPER_MODELS_CATALOG:
        is_cached = any(item["id"] in folder for folder in cached_folders)
        models.append({**item, "downloaded": is_cached})

    return {
        "whisper_models": models,
        "ocr_engines": [
            {
                "id": "rapidocr",
                "name": "RapidOCR (ONNX PP-OCRv4 - Trung/Anh/Nhật/Hàn)",
                "local": True,
            },
            {
                "id": "vietocr",
                "name": "RapidOCR + VietOCR (Chuyên trị dấu Tiếng Việt)",
                "local": True,
            },
            {
                "id": "gemini",
                "name": "Gemini 2.5 Flash Vision OCR (Cloud API)",
                "local": False,
            },
        ],
    }
