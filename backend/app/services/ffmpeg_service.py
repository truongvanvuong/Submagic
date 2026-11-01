"""FFmpeg, PyAV, and OpenCV media service for video probing, audio extraction, and frame cropping."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.schemas import ROIConfig, VideoMetadata
from app.services.hardware import resolve_ffmpeg_executable


class MediaProcessingError(RuntimeError):
    """Raised when video probing or audio extraction fails."""


def resolve_media_path(video_path: str | Path) -> Path:
    """Resolve media file path across Windows (D:\\...) and WSL (/mnt/d/...) seamlessly."""
    raw = str(video_path).strip()
    p = Path(raw)
    if p.exists() and p.is_file():
        return p.resolve()

    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw.replace("\\", "/"))
        if m:
            win_path = Path(f"{m.group(1).upper()}:/{m.group(2)}")
            if win_path.exists() and win_path.is_file():
                return win_path.resolve()
    else:
        m = re.match(r"^([a-zA-Z]):[\\/](.*)$", raw)
        if m:
            rest = m.group(2).replace("\\", "/")
            wsl_path = Path(f"/mnt/{m.group(1).lower()}/{rest}")
            if wsl_path.exists() and wsl_path.is_file():
                return wsl_path.resolve()

    fallback = settings.projects_dir / "uploads" / Path(raw.replace("\\", "/")).name
    if fallback.exists() and fallback.is_file():
        return fallback.resolve()

    return p.resolve()


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Inspect video and audio stream metadata using PyAV with OpenCV fallback."""
    path = resolve_media_path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    file_size = path.stat().st_size
    duration_sec = 0.0
    fps = 30.0
    frame_count = 0
    width = 0
    height = 0
    video_codec: str | None = None
    has_audio = False
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None

    # Primary probe via PyAV (fast, accurate container inspection including audio stream)
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is not None:
                duration_sec = float(container.duration) / float(av.time_base)

            if container.streams.video:
                v_stream = container.streams.video[0]
                width = int(v_stream.codec_context.width or v_stream.width or 0)
                height = int(v_stream.codec_context.height or v_stream.height or 0)
                video_codec = v_stream.codec_context.name
                if v_stream.average_rate:
                    fps = float(v_stream.average_rate)
                elif v_stream.base_rate:
                    fps = float(v_stream.base_rate)
                if duration_sec <= 0 and v_stream.duration and v_stream.time_base:
                    duration_sec = float(v_stream.duration * v_stream.time_base)
                if v_stream.frames and v_stream.frames > 0:
                    frame_count = int(v_stream.frames)

            if container.streams.audio:
                has_audio = True
                a_stream = container.streams.audio[0]
                audio_codec = a_stream.codec_context.name
                audio_sample_rate = int(a_stream.codec_context.sample_rate or 16000)
                audio_channels = int(a_stream.codec_context.channels or 1)
    except Exception:
        pass

    # Fallback / supplement via OpenCV if dimensions or duration are missing
    if width <= 0 or height <= 0 or duration_sec <= 0:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise MediaProcessingError(f"Cannot open video file: {path}")
        try:
            if width <= 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            if height <= 0:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if cv_fps > 0:
                fps = cv_fps
            cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if cv_frames > 0:
                frame_count = cv_frames
            if duration_sec <= 0 and fps > 0 and frame_count > 0:
                duration_sec = frame_count / fps
        finally:
            cap.release()

    if frame_count <= 0 and duration_sec > 0 and fps > 0:
        frame_count = int(round(duration_sec * fps))

    return VideoMetadata(
        file_path=str(path),
        file_name=path.name,
        file_size_bytes=file_size,
        duration_sec=round(max(0.0, duration_sec), 3),
        fps=round(max(1.0, fps), 3),
        frame_count=max(0, frame_count),
        width=width,
        height=height,
        video_codec=video_codec,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
    )


def _extract_audio_via_pyav(video_path: Path, output_wav_path: Path) -> Path:
    """Fallback audio extractor using PyAV resampler (16kHz Mono PCM S16LE)."""
    import av

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            raise MediaProcessingError(f"Video has no audio stream: {video_path.name}")
        audio_stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(16000)

            for frame in container.decode(audio_stream):
                frame.pts = None
                resampled_frames = resampler.resample(frame)
                for r_frame in resampled_frames:
                    pcm_bytes = bytes(r_frame.planes[0])
                    wav_file.writeframes(pcm_bytes)

            # Flush resampler
            flushed = resampler.resample(None)
            for r_frame in flushed:
                wav_file.writeframes(bytes(r_frame.planes[0]))

    return output_wav_path


def extract_audio_for_stt(
    video_path: str | Path,
    output_wav_path: str | Path | None = None,
    normalize_audio: bool = True,
    force_overwrite: bool = False,
) -> Path:
    """
    Extract and preprocess audio from video into 16kHz Mono 16-bit PCM WAV.
    Applies vocal frequency bandpass (80Hz - 8000Hz) + dynamic audio normalization
    when FFmpeg CLI is available; automatically falls back to PyAV decoding if needed.
    """
    src_path = resolve_media_path(video_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source media file not found: {src_path}")

    # If the input is already a 16kHz WAV file and no normalization requested, check directly
    if output_wav_path is None:
        path_hash = hashlib.md5(
            f"{src_path}_{src_path.stat().st_mtime_ns}_{normalize_audio}".encode("utf-8")
        ).hexdigest()[:12]
        out_path = settings.audio_dir / f"{src_path.stem}_{path_hash}_16k.wav"
    else:
        out_path = Path(output_wav_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024 and not force_overwrite:
        return out_path

    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
        ]
        if normalize_audio:
            # Lightweight vocal bandpass + dynamic normalization to suppress rumble/hiss
            cmd.extend(["-af", "highpass=f=80,lowpass=f=8000,dynaudnorm=p=0.9:s=5"])
        cmd.append(str(out_path))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                check=False,
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44:
                return out_path
        except Exception:
            pass

    # Fallback to PyAV native extraction
    return _extract_audio_via_pyav(src_path, out_path)


def crop_and_mask_roi(
    frame_bgr: np.ndarray,
    roi_config: ROIConfig,
) -> np.ndarray:
    """
    Crop the main ROI from a full BGR frame and paint black rectangles over any
    Exclusion Masks (watermarks/logos) overlapping the ROI.
    """
    h, w = frame_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = roi_config.main_roi.to_pixel_rect(w, h)
    cropped = frame_bgr[ry1:ry2, rx1:rx2].copy()

    for mask in roi_config.exclusion_masks:
        mx1, my1, mx2, my2 = mask.to_pixel_rect(w, h)
        # Convert full-frame mask coordinates to ROI-relative coordinates
        ix1 = max(0, mx1 - rx1)
        iy1 = max(0, my1 - ry1)
        ix2 = min(rx2 - rx1, mx2 - rx1)
        iy2 = min(ry2 - ry1, my2 - ry1)
        if ix2 > ix1 and iy2 > iy1:
            cropped[iy1:iy2, ix1:ix2] = 0

    return cropped


def extract_frame_jpeg(
    video_path: str | Path,
    timestamp_sec: float,
    roi_config: ROIConfig | None = None,
    quality: int = 90,
) -> bytes:
    """Extract a single frame at `timestamp_sec` (optionally cropped to ROI) as JPEG bytes."""
    path = resolve_media_path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaProcessingError(f"Failed to open video: {path}")

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Try first frame fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise MediaProcessingError(f"Could not decode frame at {timestamp_sec}s")

        if roi_config is not None:
            frame = crop_and_mask_roi(frame, roi_config)

        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok_enc:
            raise MediaProcessingError("Failed to encode frame to JPEG")
        return buf.tobytes()
    finally:
        cap.release()
