"""FFmpeg, PyAV, and OpenCV media service for video probing, audio extraction, and frame cropping."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.schemas import MAX_PAD_RATIO, ROIConfig, VideoMetadata
from app.services.hardware import resolve_ffmpeg_executable

logger = logging.getLogger(__name__)


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
    r_frame_rate: str | None = None
    avg_frame_rate: str | None = None
    time_base: str | None = None
    stream_start_time: float = 0.0

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
                if v_stream.base_rate:
                    r_frame_rate = (
                        f"{v_stream.base_rate.numerator}/{v_stream.base_rate.denominator}"
                        if hasattr(v_stream.base_rate, "numerator")
                        else str(v_stream.base_rate)
                    )
                if v_stream.average_rate:
                    avg_frame_rate = (
                        f"{v_stream.average_rate.numerator}/{v_stream.average_rate.denominator}"
                        if hasattr(v_stream.average_rate, "numerator")
                        else str(v_stream.average_rate)
                    )
                if v_stream.time_base:
                    time_base = (
                        f"{v_stream.time_base.numerator}/{v_stream.time_base.denominator}"
                        if hasattr(v_stream.time_base, "numerator")
                        else str(v_stream.time_base)
                    )
                if v_stream.start_time is not None and v_stream.time_base:
                    stream_start_time = float(v_stream.start_time * float(v_stream.time_base))
                elif container.start_time is not None:
                    stream_start_time = float(container.start_time) / float(av.time_base)

                if v_stream.average_rate:
                    fps = float(v_stream.average_rate)
                elif v_stream.base_rate:
                    fps = float(v_stream.base_rate)
                if v_stream.duration and v_stream.time_base:
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
        r_frame_rate=r_frame_rate,
        avg_frame_rate=avg_frame_rate,
        time_base=time_base,
        stream_start_time=round(stream_start_time, 6),
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
    pad_top_ratio: float = 0.0,
    pad_bottom_ratio: float = 0.0,
    pad_left_ratio: float = 0.0,
    pad_right_ratio: float = 0.0,
) -> np.ndarray:
    """
    Crop the main ROI from a full BGR frame and paint black rectangles over any
    Exclusion Masks (watermarks/logos) overlapping the ROI.
    Optionally expands edges by pad_*_ratio to avoid clipping diacritics / border characters.
    """
    h, w = frame_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = roi_config.main_roi.to_pixel_rect(w, h)
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    if pad_top_ratio > 0.0:
        pad_t = int(roi_h * max(0.0, min(MAX_PAD_RATIO, float(pad_top_ratio))))
        ry1 = max(0, ry1 - pad_t)
    if pad_bottom_ratio > 0.0:
        pad_b = int(roi_h * max(0.0, min(MAX_PAD_RATIO, float(pad_bottom_ratio))))
        ry2 = min(h, ry2 + pad_b)
    if pad_left_ratio > 0.0:
        pad_l = int(roi_w * max(0.0, min(MAX_PAD_RATIO, float(pad_left_ratio))))
        rx1 = max(0, rx1 - pad_l)
    if pad_right_ratio > 0.0:
        pad_r = int(roi_w * max(0.0, min(MAX_PAD_RATIO, float(pad_right_ratio))))
        rx2 = min(w, rx2 + pad_r)

    cropped = frame_bgr[ry1:ry2, rx1:rx2]

    if not roi_config.exclusion_masks:
        return cropped.copy()

    cropped = cropped.copy()
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


def extract_video_pts_table(video_path: str | Path, stream_index: int = 0) -> list[float]:
    """
    Extract presentation timestamps (PTS) in seconds for all video frames via PyAV packet demuxing.

    Key technical features:
    1. High-throughput packet demuxing (~40ms for 4,928 frames), avoiding expensive video decoding.
    2. Playback zero-normalization: shifts timestamps by stream_start_time so frame 0 starts at 0.0s.
    3. B-frame presentation reordering: sorts timestamps from DTS decode order into PTS presentation order.
    4. Guarantees strict monotonicity for binary search boundary refinement algorithms.
    5. Safe fallback: returns empty list [] on any container/stream/read failure so callers fall back to (frame_idx / fps).
    """
    try:
        path = resolve_media_path(video_path)
    except Exception:
        path = Path(str(video_path).strip())

    if not path.exists() or not path.is_file():
        logger.warning(f"PTS extraction failed: file not found: {video_path}")
        return []

    try:
        import av

        with av.open(str(path)) as container:
            if not container.streams.video or stream_index >= len(container.streams.video):
                logger.warning(
                    f"PTS extraction failed: no video stream at index {stream_index} in {path.name}"
                )
                return []

            v_stream = container.streams.video[stream_index]
            if not v_stream.time_base or float(v_stream.time_base) <= 0:
                logger.warning(f"PTS extraction failed: invalid time_base in {path.name}")
                return []

            tb = float(v_stream.time_base)

            # Compute stream start time in seconds for playback zero-normalization
            if v_stream.start_time is not None:
                stream_start_sec = float(v_stream.start_time * tb)
            elif container.start_time is not None:
                stream_start_sec = float(container.start_time) / float(av.time_base)
            else:
                stream_start_sec = 0.0

            # Demux packets from video stream
            # Demux yields packets in decode order (DTS).
            # We collect PTS and sort them to obtain presentation order (PTS) for B-frames.
            pts_raw: list[int] = []
            for packet in container.demux(v_stream):
                if packet.pts is not None:
                    pts_raw.append(packet.pts)
                elif packet.dts is not None and packet.size > 0:
                    pts_raw.append(packet.dts)

            if not pts_raw:
                logger.warning(f"PTS extraction failed: no packet timestamps found in {path.name}")
                return []

            # Reorder B-frames into presentation order
            pts_raw.sort()

            # Normalize to playback zero and microsecond precision
            pts_table = [round(max(0.0, float(p * tb) - stream_start_sec), 6) for p in pts_raw]

            # Enforce strict monotonicity if jitter or duplicate PTS exists in edge-case streams
            for i in range(len(pts_table) - 1):
                if pts_table[i + 1] <= pts_table[i]:
                    pts_table[i + 1] = round(pts_table[i] + 1e-6, 6)

            return pts_table

    except Exception as e:
        logger.warning(f"PyAV PTS extraction exception on {video_path}: {e}")
        return []


@lru_cache(maxsize=1)
def get_ffmpeg_passthrough_args(ffmpeg_exe: str | None = None) -> list[str]:
    """
    Return appropriate FFmpeg passthrough CLI arguments depending on FFmpeg version.
    - FFmpeg >= 5.1: ['-fps_mode', 'passthrough']
    - FFmpeg < 5.1 (legacy): ['-vsync', '0']
    Cached per process lifetime so version probing executes only once.
    """
    if ffmpeg_exe is None:
        ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        return ["-fps_mode", "passthrough"]

    try:
        res = subprocess.run(
            [ffmpeg_exe, "-version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            match = re.search(r"ffmpeg version (\d+)\.(\d+)", res.stdout)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if (major, minor) < (5, 1):
                    return ["-vsync", "0"]
                return ["-fps_mode", "passthrough"]
            if "fps_mode" in res.stdout:
                return ["-fps_mode", "passthrough"]
    except Exception:
        pass

    return ["-fps_mode", "passthrough"]


def get_timestamp_for_frame(
    frame_idx: int, pts_table: list[float] | None, fallback_fps: float = 25.0
) -> float:
    """
    Safely retrieve the presentation timestamp (in seconds) for a given frame index.

    Guarantees:
    - O(1) direct lookup if 0 <= frame_idx < len(pts_table).
    - Monotonic smooth linear extrapolation if frame_idx >= len(pts_table) (no IndexError).
    - Returns 0.0 if frame_idx <= 0.
    - Fallback to frame_idx / fallback_fps if pts_table is None or empty.
    """
    fps = max(1.0, float(fallback_fps))
    if frame_idx <= 0:
        return 0.0

    if not pts_table:
        return round(frame_idx / fps, 3)

    if frame_idx < len(pts_table):
        return round(pts_table[frame_idx], 3)

    # Safe extrapolation for boundary frames (e.g. last_text_idx + 1 at the end of video)
    last_pts = pts_table[-1]
    step = (
        (pts_table[-1] - pts_table[0]) / max(1, len(pts_table) - 1)
        if len(pts_table) > 1
        else (1.0 / fps)
    )
    extra_frames = frame_idx - (len(pts_table) - 1)
    return round(last_pts + extra_frames * step, 3)


class VideoROIStream:
    """
    High-throughput video stream reader yielding sequentially cropped BGR frames (ROI + exclusion masks).
    Uses FFmpeg CLI rawvideo pipe for hardware/SIMD cropped decoding without full-frame
    RGB memory allocation, falling back seamlessly to OpenCV VideoCapture.
    """

    def __init__(
        self,
        video_path: str | Path,
        roi_config: ROIConfig,
        meta: VideoMetadata | None = None,
        pad_top_ratio: float = 0.0,
        pad_bottom_ratio: float = 0.0,
        pad_left_ratio: float = 0.0,
        pad_right_ratio: float = 0.0,
        pts_table: list[float] | None = None,
    ) -> None:
        self.path = resolve_media_path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self.roi_config = roi_config
        self.meta = meta or probe_video(self.path)
        self.fps = max(1.0, float(self.meta.fps or 25.0))
        self.video_w = int(self.meta.width or 1280)
        self.video_h = int(self.meta.height or 720)
        self.pad_top_ratio = float(pad_top_ratio)
        self.pad_bottom_ratio = float(pad_bottom_ratio)
        self.pad_left_ratio = float(pad_left_ratio)
        self.pad_right_ratio = float(pad_right_ratio)
        self.pts_table = pts_table

        # Compute ROI bounding rectangle in pixels
        rx1, ry1, rx2, ry2 = self.roi_config.main_roi.to_pixel_rect(self.video_w, self.video_h)
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1

        # Allow safety margin padding clamped to [0.0, MAX_PAD_RATIO]
        if pad_top_ratio > 0.0:
            pad_t = int(roi_h * max(0.0, min(MAX_PAD_RATIO, float(pad_top_ratio))))
            ry1 = max(0, ry1 - pad_t)
        if pad_bottom_ratio > 0.0:
            pad_b = int(roi_h * max(0.0, min(MAX_PAD_RATIO, float(pad_bottom_ratio))))
            ry2 = min(self.video_h, ry2 + pad_b)
        if pad_left_ratio > 0.0:
            pad_l = int(roi_w * max(0.0, min(MAX_PAD_RATIO, float(pad_left_ratio))))
            rx1 = max(0, rx1 - pad_l)
        if pad_right_ratio > 0.0:
            pad_r = int(roi_w * max(0.0, min(MAX_PAD_RATIO, float(pad_right_ratio))))
            rx2 = min(self.video_w, rx2 + pad_r)

        # Ensure even width & height for YUV/BGR codecs in FFmpeg
        rw = max(16, (rx2 - rx1) & ~1)
        rh = max(16, (ry2 - ry1) & ~1)
        self.rx1 = max(0, min(rx1, self.video_w - rw))
        self.ry1 = max(0, min(ry1, self.video_h - rh))
        self.rw, self.rh = rw, rh

        # Precompute exclusion mask boxes relative to the cropped ROI
        self.ex_rects: list[tuple[int, int, int, int]] = []
        for em in self.roi_config.exclusion_masks:
            mx1, my1, mx2, my2 = em.to_pixel_rect(self.video_w, self.video_h)
            ix1 = max(0, mx1 - rx1)
            iy1 = max(0, my1 - ry1)
            ix2 = min(rw, mx2 - rx1)
            iy2 = min(rh, my2 - ry1)
            if ix2 > ix1 and iy2 > iy1:
                self.ex_rects.append((ix1, iy1, ix2, iy2))

        self.proc: subprocess.Popen | None = None
        self.cap: cv2.VideoCapture | None = None
        self.frame_idx = 0
        self.frame_bytes = rw * rh * 3

        # Validate video can be opened (and honor cv2.VideoCapture mocking in unit tests)
        check_cap = cv2.VideoCapture(str(self.path))
        if not check_cap.isOpened():
            raise MediaProcessingError(f"Cannot open video file: {self.path}")
        check_cap.release()

        # Try initializing FFmpeg pipe first
        ffmpeg_exe = resolve_ffmpeg_executable()
        if ffmpeg_exe:
            passthrough_flags = get_ffmpeg_passthrough_args(ffmpeg_exe)
            env_ff_threads = os.environ.get("SUBMAGIC_FFMPEG_THREADS", "").strip()
            ffmpeg_threads = env_ff_threads if (env_ff_threads.isdigit() and int(env_ff_threads) >= 1) else "2"
            cmd = [
                ffmpeg_exe,
                "-nostdin",
                "-threads",
                ffmpeg_threads,
                "-i",
                str(self.path),
                *passthrough_flags,
                "-vf",
                f"crop={rw}:{rh}:{rx1}:{ry1}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-loglevel",
                "error",
                "-",
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=max(10**6, self.frame_bytes * 10),
                )
            except Exception:
                self.proc = None

        # Fallback to OpenCV if FFmpeg pipe could not be started
        if self.proc is None:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise MediaProcessingError(f"Cannot open video file: {self.path}")

    def read_frame(self) -> tuple[int, float, np.ndarray] | None:
        """
        Read the next sequential frame.
        Returns (frame_index, timestamp_sec, cropped_roi_bgr) or None when stream ends.
        """
        current_idx = self.frame_idx
        timestamp_sec = get_timestamp_for_frame(current_idx, self.pts_table, self.fps)

        if self.proc is not None and self.proc.stdout is not None:
            raw_bytes = self.proc.stdout.read(self.frame_bytes)
            if len(raw_bytes) < self.frame_bytes:
                return None
            roi = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.rh, self.rw, 3))
            if self.ex_rects:
                roi = roi.copy()
                for ix1, iy1, ix2, iy2 in self.ex_rects:
                    roi[iy1:iy2, ix1:ix2] = 0
            self.frame_idx += 1
            return current_idx, timestamp_sec, roi

        if self.cap is not None:
            ok, full_frame = self.cap.read()
            if not ok or full_frame is None:
                return None
            fh, fw = full_frame.shape[:2]
            if fw == self.video_w and fh == self.video_h:
                roi = full_frame[self.ry1 : self.ry1 + self.rh, self.rx1 : self.rx1 + self.rw]
                if self.ex_rects:
                    roi = roi.copy()
                    for ix1, iy1, ix2, iy2 in self.ex_rects:
                        roi[iy1:iy2, ix1:ix2] = 0
            else:
                roi = crop_and_mask_roi(
                    full_frame,
                    self.roi_config,
                    pad_top_ratio=self.pad_top_ratio,
                    pad_bottom_ratio=self.pad_bottom_ratio,
                    pad_left_ratio=self.pad_left_ratio,
                    pad_right_ratio=self.pad_right_ratio,
                )
            self.frame_idx += 1
            return current_idx, timestamp_sec, roi

        return None

    def close(self) -> None:
        """Clean up process or video capture resources."""
        if self.proc is not None:
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            self.proc = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __enter__(self) -> VideoROIStream:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
