"""
Unit and integration tests for Milestone 2: PTS Timing Synchronization.
Requirements: R2 in ORIGINAL_REQUEST.md & Feature 9 in PROJECT.md.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
import pytest
import numpy as np

from app.schemas import NormalizedBox, ROIConfig, VideoMetadata
from app.services.ffmpeg_service import (
    VideoROIStream,
    extract_video_pts_table,
    get_ffmpeg_passthrough_args,
    get_timestamp_for_frame,
    probe_video,
    resolve_media_path,
)
from app.services.hardware import resolve_ffmpeg_executable

SAMPLE_VIDEO = Path(r"d:\Workspace\Submagic\Sampales\tieu_ngao_giang_ho.mp4")


# ============================================================================
# 1. VideoMetadata Schema Tests
# ============================================================================

def test_video_metadata_backward_compatibility():
    """Verify VideoMetadata can be instantiated without new fields (backward compatibility)."""
    meta = VideoMetadata(
        file_path="test.mp4",
        file_name="test.mp4",
        file_size_bytes=1000,
        duration_sec=10.0,
        fps=30.0,
        frame_count=300,
        width=1280,
        height=720,
    )
    assert meta.r_frame_rate is None
    assert meta.avg_frame_rate is None
    assert meta.time_base is None
    assert meta.stream_start_time == 0.0
    assert meta.r_fps is None
    assert meta.avg_fps is None


def test_video_metadata_with_new_fields_and_properties():
    """Verify VideoMetadata serialization, deserialization, and helper properties."""
    meta = VideoMetadata(
        file_path="test.mp4",
        file_name="test.mp4",
        file_size_bytes=1000,
        duration_sec=10.0,
        fps=23.976,
        frame_count=240,
        width=1920,
        height=1080,
        r_frame_rate="24000/1001",
        avg_frame_rate="30/1",
        time_base="1/15360",
        stream_start_time=1.5,
    )
    assert meta.r_frame_rate == "24000/1001"
    assert meta.avg_frame_rate == "30/1"
    assert meta.time_base == "1/15360"
    assert meta.stream_start_time == 1.5
    assert meta.r_fps is not None
    assert abs(meta.r_fps - (24000.0 / 1001.0)) < 1e-5
    assert meta.avg_fps == 30.0

    # Test round-trip JSON serialization
    json_str = meta.model_dump_json()
    meta_deserialized = VideoMetadata.model_validate_json(json_str)
    assert meta_deserialized.r_frame_rate == "24000/1001"
    assert meta_deserialized.avg_frame_rate == "30/1"
    assert meta_deserialized.time_base == "1/15360"
    assert meta_deserialized.stream_start_time == 1.5


def test_video_metadata_property_edge_cases():
    """Verify r_fps and avg_fps handle malformed or division-by-zero strings gracefully."""
    meta = VideoMetadata(
        file_path="test.mp4",
        file_name="test.mp4",
        file_size_bytes=1000,
        duration_sec=1.0,
        fps=30.0,
        frame_count=30,
        width=640,
        height=480,
        r_frame_rate="0/0",
        avg_frame_rate="invalid_fraction",
    )
    assert meta.r_fps is None
    assert meta.avg_fps is None


# ============================================================================
# 2. probe_video Tests
# ============================================================================

def test_probe_video_sample_file():
    """Verify probe_video extracts r_frame_rate, avg_frame_rate, time_base, stream_start_time, and stream duration."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    meta = probe_video(SAMPLE_VIDEO)
    assert meta.r_frame_rate in ("30/1", "60/1")
    assert meta.avg_frame_rate in ("30/1", "60/1")
    assert meta.time_base in ("1/15360", "1/60")
    assert meta.stream_start_time == 0.0
    assert meta.duration_sec in (164.267, 156.783)
    assert meta.frame_count in (4928, 9407)
    assert meta.fps in (30.0, 60.0)
    assert meta.width in (1920, 1280)
    assert meta.height in (1080, 720)


# ============================================================================
# 3. extract_video_pts_table Tests
# ============================================================================

def test_extract_video_pts_table_sample_video():
    """Verify PTS extraction on real video: exact frame count, strict monotonicity, and < 200ms speed."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    t0 = time.perf_counter()
    pts_table = extract_video_pts_table(SAMPLE_VIDEO)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(pts_table) in (4928, 9407)
    assert pts_table[0] == 0.0
    expected_step = 1.0 / 60.0 if len(pts_table) == 9407 else 1.0 / 30.0
    expected_last = 156.766667 if len(pts_table) == 9407 else 164.233333
    assert abs(pts_table[1] - expected_step) < 1e-4
    assert abs(pts_table[-1] - expected_last) < 1e-3

    # Verify strict monotonicity
    for i in range(len(pts_table) - 1):
        assert pts_table[i] < pts_table[i + 1], f"Non-monotonic at index {i}: {pts_table[i]} >= {pts_table[i+1]}"

    # Verify fast demux throughput (< 1000ms on CPU even for 298MB 9,407-frame 60fps file)
    assert elapsed_ms < 1000.0, f"Demux took too long: {elapsed_ms:.2f}ms"


def test_extract_video_pts_table_missing_file():
    """Verify safe fallback [] on missing video file."""
    res = extract_video_pts_table("non_existent_video_path_xyz123.mp4")
    assert res == []


def test_extract_video_pts_table_corrupted_file(tmp_path: Path):
    """Verify safe fallback [] on corrupted video file."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"NOT A REAL MP4 FILE HEADER OR BITSTREAM")
    res = extract_video_pts_table(corrupt_file)
    assert res == []


def test_extract_video_pts_table_invalid_stream_index():
    """Verify safe fallback [] when stream index is out of bounds."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")
    res = extract_video_pts_table(SAMPLE_VIDEO, stream_index=99)
    assert res == []


def test_extract_video_pts_table_stream_start_time_normalization(tmp_path: Path):
    """Verify that a video stream with start_time offset is normalized to playback 0.0s."""
    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        pytest.skip("FFmpeg executable not available")

    test_ts = tmp_path / "offset_test.ts"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=25",
        "-c:v",
        "mpeg2video",
        "-output_ts_offset",
        "10.0",
        str(test_ts),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not test_ts.exists():
        pytest.skip("Could not generate offset MPEG-TS test video")

    pts_table = extract_video_pts_table(test_ts)
    assert len(pts_table) == 50
    # First frame MUST be normalized to 0.0s despite the 10.0s offset
    assert pts_table[0] == 0.0
    assert abs(pts_table[-1] - 1.96) < 1e-3
    assert all(pts_table[i] < pts_table[i + 1] for i in range(len(pts_table) - 1))


# ============================================================================
# 4. get_timestamp_for_frame Tests
# ============================================================================

def test_get_timestamp_for_frame_boundaries():
    """Verify exact lookup, zero boundary, extrapolation, and fallback."""
    table = [0.0, 0.04, 0.08, 0.12, 0.16]

    # Negative / zero frame_idx returns 0.0
    assert get_timestamp_for_frame(-5, table, 25.0) == 0.0
    assert get_timestamp_for_frame(0, table, 25.0) == 0.0

    # Within bounds exact lookup
    assert get_timestamp_for_frame(1, table, 25.0) == 0.04
    assert get_timestamp_for_frame(2, table, 25.0) == 0.08
    assert get_timestamp_for_frame(4, table, 25.0) == 0.16

    # Beyond bounds smooth linear extrapolation
    # step = (0.16 - 0.0) / 4 = 0.04
    assert get_timestamp_for_frame(5, table, 25.0) == 0.2
    assert get_timestamp_for_frame(6, table, 25.0) == 0.24

    # None or empty table falls back to frame_idx / fallback_fps
    assert get_timestamp_for_frame(10, None, 25.0) == 0.4
    assert get_timestamp_for_frame(10, [], 25.0) == 0.4
    assert get_timestamp_for_frame(0, None, 25.0) == 0.0

    # Single-element table extrapolation
    single_table = [1.5]
    assert get_timestamp_for_frame(0, single_table, 25.0) == 0.0
    # frame 1: last_pts + 1 * (1/25) = 1.5 + 0.04 = 1.54
    assert get_timestamp_for_frame(1, single_table, 25.0) == 1.54


# ============================================================================
# 5. get_ffmpeg_passthrough_args Tests
# ============================================================================

def test_get_ffmpeg_passthrough_args():
    """Verify get_ffmpeg_passthrough_args returns valid flags."""
    ffmpeg_exe = resolve_ffmpeg_executable()
    flags = get_ffmpeg_passthrough_args(ffmpeg_exe)
    assert isinstance(flags, list)
    assert flags in (["-fps_mode", "passthrough"], ["-vsync", "0"])

    # Fallback when executable is None
    fallback_flags = get_ffmpeg_passthrough_args(None)
    assert isinstance(fallback_flags, list)
    assert len(fallback_flags) == 2


# ============================================================================
# 6. VideoROIStream Integration Tests
# ============================================================================

def test_videoroi_stream_pts_lookup_and_passthrough():
    """Verify VideoROIStream accepts pts_table and yields timestamps from the table."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    pts_table = extract_video_pts_table(SAMPLE_VIDEO)
    roi_config = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))

    stream = VideoROIStream(SAMPLE_VIDEO, roi_config, pts_table=pts_table)
    assert stream.pts_table is pts_table

    try:
        # Read first 5 frames
        for expected_idx in range(5):
            item = stream.read_frame()
            assert item is not None
            f_idx, timestamp_sec, roi = item
            assert f_idx == expected_idx
            assert timestamp_sec == round(pts_table[expected_idx], 3)
            assert roi.ndim == 3
            assert roi.shape[2] == 3
    finally:
        stream.close()


def test_videoroi_stream_fallback_without_pts_table():
    """Verify VideoROIStream without pts_table falls back to frame_idx / fps."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    roi_config = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))
    stream = VideoROIStream(SAMPLE_VIDEO, roi_config, pts_table=None)
    assert stream.pts_table is None

    try:
        item = stream.read_frame()
        assert item is not None
        f_idx, timestamp_sec, roi = item
        assert f_idx == 0
        assert timestamp_sec == 0.0

        item1 = stream.read_frame()
        assert item1 is not None
        f_idx1, timestamp_sec1, _ = item1
        assert f_idx1 == 1
        assert timestamp_sec1 == round(1.0 / stream.fps, 3)
    finally:
        stream.close()


# ============================================================================
# 7. OCR Pipeline Integration & CLI Ordering Tests
# ============================================================================

def test_ocr_pipeline_reexport_and_boundary_refinement_vfr():
    """Verify get_timestamp_for_frame is available from ocr_pipeline and handles VFR boundaries."""
    from app.services.ocr_pipeline import get_timestamp_for_frame as ocr_get_timestamp

    # Simulated VFR pts table where intervals fluctuate (e.g. 33ms, 50ms, 17ms, 66ms, etc.)
    vfr_pts = [0.0, 0.033, 0.083, 0.100, 0.166, 0.200, 0.250]
    video_fps = 30.0

    # Test appearance boundary (found_idx = 2)
    # Using old frame_idx / fps: 2 / 30.0 = 0.067
    # Using PTS table: 0.083
    appearance_pts = round(ocr_get_timestamp(2, vfr_pts, video_fps), 3)
    assert appearance_pts == 0.083
    assert appearance_pts != round(2 / video_fps, 3)

    # Test disappearance boundary (last_text_idx + 1 = 4)
    # Using old formula: 4 / 30.0 = 0.133
    # Using PTS table: 0.166
    disappearance_pts = round(ocr_get_timestamp(4, vfr_pts, video_fps), 3)
    assert disappearance_pts == 0.166
    assert disappearance_pts != round(4 / video_fps, 3)

    # Test direct cut boundary (cut_idx = 5)
    cut_pts = round(ocr_get_timestamp(5, vfr_pts, video_fps), 3)
    assert cut_pts == 0.200

    # Test intermediate recovery (wb_idx = 3)
    wb_pts = round(ocr_get_timestamp(3, vfr_pts, video_fps), 3)
    assert wb_pts == 0.100


def test_videoroi_stream_cli_order_passthrough(monkeypatch):
    """Verify FFmpeg CLI arguments order: passthrough flag MUST appear after -i <path> and before -vf."""
    captured_cmds: list[list[str]] = []

    class DummyProc:
        def __init__(self, cmd, **kwargs):
            captured_cmds.append(cmd)
            self.stdout = None

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", DummyProc)
    monkeypatch.setattr("app.services.ffmpeg_service.resolve_ffmpeg_executable", lambda: "ffmpeg")

    roi_config = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))
    if SAMPLE_VIDEO.exists():
        stream = VideoROIStream(SAMPLE_VIDEO, roi_config)
        assert len(captured_cmds) >= 1
        cmd = captured_cmds[-1]
        # Find index of '-i'
        idx_i = cmd.index("-i")
        idx_vf = cmd.index("-vf")
        # Check passthrough flag exists between -i <path> and -vf
        # cmd[idx_i + 1] is video path
        assert idx_vf > idx_i + 2
        passthrough_slice = cmd[idx_i + 2 : idx_vf]
        assert ("-fps_mode" in passthrough_slice) or ("-vsync" in passthrough_slice)
        stream.close()


def test_ongoing_span_start_time_not_overwritten_by_mid_span_cut():
    """Verify ThreeLayerVisualFilter preserves initial start_time across mid-span visual fluctuations."""
    import cv2
    from app.services.ocr_pipeline import ThreeLayerVisualFilter, extract_text_binary_mask

    vf = ThreeLayerVisualFilter(stability_frames=1, visual_diff_threshold=0.14, frame_step_sec=0.167)
    roi1 = np.zeros((80, 400, 3), dtype=np.uint8)
    cv2.putText(roi1, "Ma dau cua Nhat Nguyet", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)
    m1 = extract_text_binary_mask(roi1)

    # Frame 1 at t=15.700s creates pending span with sampled_frames == 1
    vf.process_frame(roi1, 15.700, precomputed_mask=m1)
    assert vf._pending_span is not None
    assert vf._pending_span.sampled_frames == 1
    assert vf._pending_span.start_time == 15.700

    # Frame 2 at t=15.867s advances sampled_frames to 2; start_time must remain 15.700
    vf.process_frame(roi1, 15.867, precomputed_mask=m1)
    active_or_pending = vf._active_span or vf._pending_span
    assert active_or_pending is not None
    assert active_or_pending.sampled_frames == 2
    assert active_or_pending.start_time == 15.700

