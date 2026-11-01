"""Hardware and runtime environment detection service."""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache

from app.schemas import HardwareInfo


def _get_windows_ram_gb() -> float:
    """Retrieve total physical RAM in GB on Windows via Win32 API without extra deps."""
    if platform.system() != "Windows":
        return 8.0
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        return 8.0


def _detect_nvidia_gpus() -> list[str]:
    """Query nvidia-smi for GPU names if available."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        default_win_path = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.exists(default_win_path):
            nvidia_smi = default_win_path
    if not nvidia_smi:
        return []

    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    except Exception:
        pass
    return []


def resolve_ffmpeg_executable() -> str | None:
    """Locate ffmpeg executable from system PATH or bundled imageio-ffmpeg."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareInfo:
    """Detect CPU, RAM, GPU, CUDA, ONNX Runtime providers, and FFmpeg."""
    os_name = f"{platform.system()} {platform.release()}"
    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    cpu_cores = os.cpu_count() or 4
    ram_total_gb = _get_windows_ram_gb()

    cuda_device_count = 0
    try:
        import ctranslate2
        cuda_device_count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        cuda_device_count = 0

    gpu_names = _detect_nvidia_gpus()
    cuda_available = cuda_device_count > 0

    onnx_providers: list[str] = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        onnx_providers = list(ort.get_available_providers())
    except Exception:
        pass

    if cuda_available:
        recommended_stt_device = "cuda"
        recommended_stt_compute_type = "float16"
    else:
        recommended_stt_device = "cpu"
        recommended_stt_compute_type = "int8"

    if "CUDAExecutionProvider" in onnx_providers:
        recommended_ocr_provider = "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in onnx_providers:
        recommended_ocr_provider = "DmlExecutionProvider"
    else:
        recommended_ocr_provider = "CPUExecutionProvider"

    ffmpeg_path = resolve_ffmpeg_executable()

    return HardwareInfo(
        os_name=os_name,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        gpu_names=gpu_names,
        onnx_providers=onnx_providers,
        recommended_stt_device=recommended_stt_device,
        recommended_stt_compute_type=recommended_stt_compute_type,
        recommended_ocr_provider=recommended_ocr_provider,
        ffmpeg_available=ffmpeg_path is not None,
        ffmpeg_path=ffmpeg_path,
    )


def resolve_stt_device_and_compute(
    requested_device: str = "auto", requested_compute_type: str = "auto"
) -> tuple[str, str]:
    """Resolve optimal CTranslate2/Faster-Whisper device and compute_type."""
    hw = detect_hardware()
    if requested_device == "auto":
        device = hw.recommended_stt_device
    elif requested_device == "cuda" and not hw.cuda_available:
        device = "cpu"
    else:
        device = requested_device

    if requested_compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = requested_compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

    return device, compute_type
