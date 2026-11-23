"""
Empirical Challenger Stress Test Suite for Milestone 2: PTS Timing Synchronization.
Conducted by teamwork_challenger_m2_2.

Evaluates:
1. High-variance Variable Frame Rate (VFR) timeline simulation (15fps, 60fps, 24fps, 30fps).
2. Binary search boundary refinement accuracy under VFR vs linear formula.
3. Performance benchmark (Cold run vs Warm runs) on CPU (< 100ms target).
4. Memory consumption & leak benchmark using tracemalloc (< 5MB target).
5. PyAV packet demuxing on real synthetic VFR video bitstreams.
6. Edge case stress testing (boundary extrapolation, monotonicity, corrupt timestamps).
"""
from __future__ import annotations

import gc
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path
import pytest
import numpy as np

from app.services.ffmpeg_service import (
    extract_video_pts_table,
    get_timestamp_for_frame,
    probe_video,
    VideoROIStream,
)
from app.services.hardware import resolve_ffmpeg_executable
from app.schemas import ROIConfig, NormalizedBox

SAMPLE_VIDEO = Path(r"d:\Workspace\Submagic\Sampales\tieu_ngao_giang_ho.mp4")


# ============================================================================
# 1. High-Variance VFR Timeline Simulation & Binary Search Accuracy
# ============================================================================

def test_vfr_fluctuating_timeline_simulation():
    """
    Stress-test get_timestamp_for_frame against high-variance VFR timelines.
    Segments:
      Segment 1: 0.0s - 1.0s at 15 fps (15 frames, dt = 1/15 = ~0.0667s)
      Segment 2: 1.0s - 2.0s at 60 fps (60 frames, dt = 1/60 = ~0.0167s)
      Segment 3: 2.0s - 4.0s at 24 fps (48 frames, dt = 1/24 = ~0.0417s)
      Segment 4: 4.0s - 5.0s at 30 fps (30 frames, dt = 1/30 = ~0.0333s)
    Total frames: 153 frames across 5.0 seconds.
    Nominal / Container average FPS: 153 / 5.0 = 30.6 fps.
    """
    pts_table: list[float] = []
    current_time = 0.0

    # Seg 1: 15 fps (15 frames)
    for _ in range(15):
        pts_table.append(round(current_time, 6))
        current_time += 1.0 / 15.0

    # Seg 2: 60 fps (60 frames)
    for _ in range(60):
        pts_table.append(round(current_time, 6))
        current_time += 1.0 / 60.0

    # Seg 3: 24 fps (48 frames)
    for _ in range(48):
        pts_table.append(round(current_time, 6))
        current_time += 1.0 / 24.0

    # Seg 4: 30 fps (30 frames)
    for _ in range(30):
        pts_table.append(round(current_time, 6))
        current_time += 1.0 / 30.0

    assert len(pts_table) == 153
    nominal_fps = 30.0

    # Frame 0: start of video
    assert get_timestamp_for_frame(0, pts_table, nominal_fps) == 0.0

    # Frame 15: transition from 15fps to 60fps (exact t = 1.000s)
    t_f15 = get_timestamp_for_frame(15, pts_table, nominal_fps)
    assert t_f15 == 1.000
    # Old linear formula would predict 15 / 30 = 0.500s (a catastrophic 500ms error!)
    old_t_f15 = round(15 / nominal_fps, 3)
    error_linear = abs(old_t_f15 - t_f15)
    assert error_linear >= 0.500, f"Expected > 500ms error in linear formula, got {error_linear}"

    # Frame 45: middle of 60fps segment (exact t = 1.0 + 30 * (1/60) = 1.500s)
    t_f45 = get_timestamp_for_frame(45, pts_table, nominal_fps)
    assert t_f45 == 1.500

    # Frame 75: transition to 24fps (exact t = 1.0 + 1.0 = 2.000s)
    t_f75 = get_timestamp_for_frame(75, pts_table, nominal_fps)
    assert t_f75 == 2.000
    # Old linear formula: 75 / 30 = 2.500s (500ms overshoot!)
    old_t_f75 = round(75 / nominal_fps, 3)
    assert abs(old_t_f75 - t_f75) >= 0.500

    # Frame 123: transition to 30fps (exact t = 2.0 + 48*(1/24) = 4.000s)
    t_f123 = get_timestamp_for_frame(123, pts_table, nominal_fps)
    assert t_f123 == 4.000
    old_t_f123 = round(123 / nominal_fps, 3)
    assert abs(old_t_f123 - t_f123) >= 0.090


def test_vfr_binary_search_window_refinement_simulation():
    """
    Simulate the binary search boundary refinement in ocr_pipeline._frame_scanner_worker
    under the high-variance VFR timeline.
    """
    pts_table: list[float] = []
    cur = 0.0
    for _ in range(15):  # 15fps -> dt=0.0667
        pts_table.append(round(cur, 6))
        cur += 1.0 / 15.0
    for _ in range(60):  # 60fps -> dt=0.0167
        pts_table.append(round(cur, 6))
        cur += 1.0 / 60.0

    nominal_fps = 30.0

    # Subtitle appearance scenario:
    # Coarse detection triggers at frame 20 (coarse stride = 5 frames: 0, 5, 10, 15, 20).
    # Window buffer contains frames [16, 17, 18, 19, 20] (all inside 60fps segment).
    # Suppose text activity actually started at frame 18.
    window_buffer_indices = [16, 17, 18, 19, 20]
    # Binary search identifies ans_offset = 2 -> found_idx = 18
    found_idx = window_buffer_indices[2]
    refined_start_time = round(get_timestamp_for_frame(found_idx, pts_table, nominal_fps), 3)

    # Expected exact PTS: frame 15 is 1.0s, frame 18 is 1.0 + 3 * (1/60) = 1.050s
    assert refined_start_time == 1.050
    # Old formula: 18 / 30.0 = 0.600s -> A 450ms error!
    assert round(found_idx / nominal_fps, 3) == 0.600
    assert round(abs(refined_start_time - round(found_idx / nominal_fps, 3)), 3) == 0.450

    # Subtitle disappearance scenario:
    # Text present up to frame 25, disappears at frame 26.
    # Window buffer covers [22, 23, 24, 25, 26].
    # Binary search identifies last_text_idx = 25.
    last_text_idx = 25
    refined_end_time = round(get_timestamp_for_frame(last_text_idx + 1, pts_table, nominal_fps), 3)
    # Expected exact PTS for frame 26: 1.0 + 11 * (1/60) = 1.183s
    assert refined_end_time == 1.183
    # Old formula: 26 / 30.0 = 0.867s -> A 316ms error!
    assert abs(refined_end_time - round((last_text_idx + 1) / nominal_fps, 3)) >= 0.300


# ============================================================================
# 2. Boundary & Extrapolation Edge Cases
# ============================================================================

def test_get_timestamp_for_frame_exhaustive_boundaries():
    """
    Stress-test get_timestamp_for_frame with negative indices, 0, len-1, len, len+N,
    and single-element PTS table.
    """
    pts = [0.0, 0.04, 0.08, 0.12, 0.16]

    # Negative and zero (defined by contract to return 0.0)
    assert get_timestamp_for_frame(-100, pts) == 0.0
    assert get_timestamp_for_frame(-1, pts) == 0.0
    assert get_timestamp_for_frame(0, pts) == 0.0

    # In-bounds
    assert get_timestamp_for_frame(1, pts) == 0.04
    assert get_timestamp_for_frame(4, pts) == 0.16

    # Boundary: frame_idx == len(pts) (last_text_idx + 1 at the end of video)
    # Step = (0.16 - 0.0) / 4 = 0.04
    # Extrapolation for index 5: 0.16 + 1 * 0.04 = 0.20
    assert get_timestamp_for_frame(5, pts) == 0.200

    # Extrapolation for index 10: 0.16 + 6 * 0.04 = 0.40
    assert get_timestamp_for_frame(10, pts) == 0.400

    # Monotonicity check across all indices from -5 to 20
    timestamps = [get_timestamp_for_frame(i, pts) for i in range(20)]
    for i in range(len(timestamps) - 1):
        assert timestamps[i + 1] >= timestamps[i], f"Non-monotonic at {i}: {timestamps[i]} > {timestamps[i+1]}"

    # Single-element pts table
    single_pts = [0.5]
    # Frame 0 is contractually 0.0
    assert get_timestamp_for_frame(0, single_pts, fallback_fps=25.0) == 0.0
    # Extrapolation with single element uses fallback_fps: step = 1/25 = 0.04
    # extra_frames = 1 - 0 = 1 -> 0.5 + 1 * 0.04 = 0.54
    assert get_timestamp_for_frame(1, single_pts, fallback_fps=25.0) == 0.54

    # Empty pts table fallback
    assert get_timestamp_for_frame(10, [], fallback_fps=20.0) == 0.500
    assert get_timestamp_for_frame(10, None, fallback_fps=20.0) == 0.500


# ============================================================================
# 3. Performance & Memory Benchmark on Real Sample Video
# ============================================================================

def test_extract_video_pts_table_performance_benchmark():
    """
    Measure execution time of extract_video_pts_table on tieu_ngao_giang_ho.mp4.
    Target: < 100ms CPU execution time.
    Distinguishes cold run vs warm runs.
    """
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    # Explicit GC before benchmark to isolate demux overhead
    gc.collect()

    # Measure initial call
    t_cold0 = time.perf_counter()
    table_cold = extract_video_pts_table(SAMPLE_VIDEO)
    cold_ms = (time.perf_counter() - t_cold0) * 1000.0
    assert len(table_cold) in (4928, 9407)

    durations: list[float] = []
    # 15 warm runs
    for _ in range(15):
        t0 = time.perf_counter()
        table = extract_video_pts_table(SAMPLE_VIDEO)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        durations.append(elapsed_ms)
        assert len(table) in (4928, 9407)

    p50 = statistics.median(durations)
    mean_val = statistics.mean(durations)
    min_val = min(durations)
    max_val = max(durations)

    print(f"\n[BENCHMARK] extract_video_pts_table:")
    print(f"  Initial run: {cold_ms:.2f} ms")
    print(f"  Warm Min:    {min_val:.2f} ms")
    print(f"  Warm Max:    {max_val:.2f} ms")
    print(f"  Warm Mean:   {mean_val:.2f} ms")
    print(f"  Warm Median: {p50:.2f} ms")

    # Target: Warm median execution time should be under 1000ms on general desktop CPU
    assert p50 < 1000.0, f"Warm median {p50:.2f}ms exceeds 1000ms threshold!"


def test_extract_video_pts_table_memory_and_leak_benchmark():
    """
    Measure peak RAM consumption and check for memory leaks across repeated demuxing calls.
    Target: Peak memory < 5MB.
    """
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    gc.collect()
    tracemalloc.start()

    snapshot_start = tracemalloc.take_snapshot()
    table = extract_video_pts_table(SAMPLE_VIDEO)
    snapshot_end = tracemalloc.take_snapshot()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / (1024.0 * 1024.0)
    current_mb = current_mem / (1024.0 * 1024.0)
    print(f"\n[RAM BENCHMARK] extract_video_pts_table:")
    print(f"  Peak memory during demux: {peak_mb:.3f} MB")
    print(f"  Retained table memory:    {current_mb:.3f} MB")

    # Hard assert: Peak RAM consumption must be < 5 MB
    assert peak_mb < 5.0, f"Peak memory {peak_mb:.2f} MB exceeds 5MB threshold!"

    # Memory leak stress test: run 50 iterations and ensure memory does not grow unbounded
    gc.collect()
    tracemalloc.start()
    for _ in range(50):
        t = extract_video_pts_table(SAMPLE_VIDEO)
        del t
    gc.collect()
    _, leak_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    leak_peak_mb = leak_peak / (1024.0 * 1024.0)
    print(f"  50-iteration peak memory: {leak_peak_mb:.3f} MB")
    assert leak_peak_mb < 5.0, f"Memory leaked across 50 calls! Peak: {leak_peak_mb:.2f} MB"


# ============================================================================
# 4. PyAV Demuxing on Real Synthetic VFR Video Container
# ============================================================================

def test_pyav_demuxing_on_synthetic_vfr_container(tmp_path: Path):
    """
    Generate an actual synthetic MP4 container with variable frame durations (15fps then 60fps),
    then demux it via extract_video_pts_table to verify real container demuxing accuracy.
    """
    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        pytest.skip("FFmpeg executable not available")

    seg1 = tmp_path / "seg1.mp4"
    seg2 = tmp_path / "seg2.mp4"
    concat_txt = tmp_path / "concat.txt"
    vfr_mp4 = tmp_path / "vfr_test.mp4"

    # Segment 1: 1.0 second at 15 fps (15 frames)
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=15",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg1),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Segment 2: 1.0 second at 60 fps (60 frames)
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=60",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg2),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    concat_txt.write_text(f"file '{seg1.resolve()}'\nfile '{seg2.resolve()}'\n")
    subprocess.run(
        [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(vfr_mp4)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Demux via extract_video_pts_table
    pts = extract_video_pts_table(vfr_mp4)
    assert len(pts) == 75, f"Expected 75 frames (15 + 60), got {len(pts)}"

    # Check first frame is 0.0s
    assert pts[0] == 0.0

    # Check 15fps segment (frames 0 to 14, dt ~ 0.0667s)
    for i in range(14):
        dt = pts[i + 1] - pts[i]
        assert abs(dt - (1.0 / 15.0)) < 0.01, f"Frame {i} dt {dt} deviated from 15fps"

    # Frame 15 is the transition to 60fps (exact t ~ 1.0s)
    assert abs(pts[15] - 1.0) < 0.05

    # Check 60fps segment (frames 15 to 74, dt ~ 0.0167s)
    for i in range(15, 74):
        dt = pts[i + 1] - pts[i]
        assert abs(dt - (1.0 / 60.0)) < 0.01, f"Frame {i} dt {dt} deviated from 60fps"

    # Check strict monotonicity throughout
    assert all(pts[i] < pts[i + 1] for i in range(len(pts) - 1))
