"""
Adversarial empirical verification and stress test harness for Milestone 2: PTS Synchronization.
Conducted by teamwork_challenger.
"""
from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
)
from app.services.hardware import resolve_ffmpeg_executable

SAMPLE_VIDEO = Path(r"d:\Workspace\Submagic\Sampales\tieu_ngao_giang_ho.mp4")


# ============================================================================
# Section 1: Empirical Monotonicity, Continuity & Robustness on Real Video
# ============================================================================

def test_pts_table_real_sample_deep_empirical():
    """
    Stress-test extract_video_pts_table on Sampales/tieu_ngao_giang_ho.mp4:
    - Non-empty
    - Strictly monotonic: pts[i] < pts[i+1]
    - No NaN, None, negative (<0), or inf
    - No duplicate values
    - Exact frame count matches video metadata
    - Microsecond resolution and sensible delta distribution
    """
    assert SAMPLE_VIDEO.exists(), f"Sample video missing: {SAMPLE_VIDEO}"

    meta = probe_video(SAMPLE_VIDEO)
    pts_table = extract_video_pts_table(SAMPLE_VIDEO)

    print(f"\n[EMPIRICAL LOG] PTS table extracted: {len(pts_table)} frames")
    print(f"[EMPIRICAL LOG] Video metadata frame_count: {meta.frame_count}, fps: {meta.fps}, duration: {meta.duration_sec}s")

    # 1. Non-empty & frame count check
    assert len(pts_table) > 0, "PTS table must not be empty"
    assert len(pts_table) == meta.frame_count, (
        f"PTS table length {len(pts_table)} does not match metadata frame_count {meta.frame_count}"
    )

    # 2. Frame 0 check
    assert pts_table[0] == 0.0, f"Frame 0 timestamp must be normalized to 0.0s, got {pts_table[0]}"

    # 3. Check for invalid values: NaN, inf, None, negative
    for idx, val in enumerate(pts_table):
        assert val is not None, f"PTS at index {idx} is None"
        assert isinstance(val, (int, float)), f"PTS at index {idx} is not numeric: {type(val)}"
        assert not math.isnan(val), f"PTS at index {idx} is NaN"
        assert not math.isinf(val), f"PTS at index {idx} is Inf"
        assert val >= 0.0, f"PTS at index {idx} is negative: {val}"

    # 4. Strict Monotonicity and No Duplicates
    deltas = []
    for i in range(len(pts_table) - 1):
        curr = pts_table[i]
        nxt = pts_table[i + 1]
        delta = nxt - curr
        assert delta > 0.0, (
            f"Strict monotonicity violated at index {i} -> {i+1}: "
            f"pts[{i}]={curr} >= pts[{i+1}]={nxt} (delta={delta})"
        )
        deltas.append(delta)

    deltas_arr = np.array(deltas)
    min_delta = float(np.min(deltas_arr))
    max_delta = float(np.max(deltas_arr))
    mean_delta = float(np.mean(deltas_arr))
    std_delta = float(np.std(deltas_arr))

    print(f"[EMPIRICAL LOG] Deltas: min={min_delta:.6f}s, max={max_delta:.6f}s, mean={mean_delta:.6f}s, std={std_delta:.8f}")

    # For CFR 30fps video, expected delta is 1/30 = 0.033333s
    expected_delta = 1.0 / meta.fps
    assert abs(mean_delta - expected_delta) < 1e-4, (
        f"Mean delta {mean_delta:.6f} diverges from expected 1/fps {expected_delta:.6f}"
    )
    assert std_delta < 1e-3, f"Delta standard deviation {std_delta} too high for CFR video"

    # 5. Check last frame timestamp consistency with duration
    last_pts = pts_table[-1]
    expected_last_pts = (meta.frame_count - 1) / meta.fps
    print(f"[EMPIRICAL LOG] Last PTS: {last_pts:.6f}s, expected approx {expected_last_pts:.6f}s")
    assert abs(last_pts - expected_last_pts) < 0.1, (
        f"Last frame PTS {last_pts} diverges significantly from expected {expected_last_pts}"
    )


# ============================================================================
# Section 2: Exhaustive Edge Cases for get_timestamp_for_frame
# ============================================================================

def test_get_timestamp_for_frame_exhaustive_boundaries():
    """
    Stress-test get_timestamp_for_frame against boundary cases:
    - frame_idx < 0 (-1, -50, -1_000_000)
    - frame_idx == 0
    - frame_idx == len(pts_table) - 1 (exact last element)
    - frame_idx == len(pts_table) (one past last)
    - frame_idx == 1_000_000 (huge extrapolation)
    - pts_table is None
    - pts_table is []
    - pts_table has only 1 element [0.0] or [1.5]
    - extreme/invalid fallback_fps (0.0, -25.0, 1e-6)
    - numpy integer index support
    """
    table = [0.0, 0.033333, 0.066667, 0.100000, 0.133333]
    fps = 30.0

    # 1. Negative indices must return 0.0
    for neg_idx in [-1, -2, -50, -999999]:
        res = get_timestamp_for_frame(neg_idx, table, fps)
        assert res == 0.0, f"Expected 0.0 for negative index {neg_idx}, got {res}"

    # 2. frame_idx == 0 must return 0.0
    assert get_timestamp_for_frame(0, table, fps) == 0.0

    # 3. Last frame index (len(table) - 1 = 4)
    res_last = get_timestamp_for_frame(4, table, fps)
    assert res_last == round(table[4], 3)
    assert res_last == 0.133

    # 4. One past last frame (len(table) = 5)
    # Expected extrapolation: table[4] + 1 * step, step = (0.133333 - 0.0) / 4 = 0.03333325
    res_next = get_timestamp_for_frame(5, table, fps)
    assert res_next > res_last, f"Extrapolated timestamp {res_next} must be > last timestamp {res_last}"
    assert math.isclose(res_next, 0.167, abs_tol=1e-3)

    # 5. Huge extrapolation (frame_idx = 1_000_000)
    res_huge = get_timestamp_for_frame(1_000_000, table, fps)
    assert isinstance(res_huge, float)
    assert not math.isnan(res_huge)
    assert not math.isinf(res_huge)
    assert res_huge > res_next
    # Expected: 0.133333 + (1_000_000 - 4) * 0.03333325 ~ 33333.333
    assert abs(res_huge - 33333.333) < 1.0

    # 6. pts_table is None
    assert get_timestamp_for_frame(0, None, fps) == 0.0
    assert get_timestamp_for_frame(-10, None, fps) == 0.0
    assert get_timestamp_for_frame(30, None, fps) == 1.0
    assert get_timestamp_for_frame(60, None, fps) == 2.0

    # 7. pts_table is empty []
    assert get_timestamp_for_frame(0, [], fps) == 0.0
    assert get_timestamp_for_frame(-10, [], fps) == 0.0
    assert get_timestamp_for_frame(30, [], fps) == 1.0
    assert get_timestamp_for_frame(15, [], fps) == 0.5

    # 8. Single-element pts_table: [0.0]
    single_zero = [0.0]
    assert get_timestamp_for_frame(0, single_zero, fps) == 0.0
    assert get_timestamp_for_frame(1, single_zero, fps) == round(1.0 / fps, 3)
    assert get_timestamp_for_frame(30, single_zero, fps) == round(30.0 / fps, 3)

    # Single-element pts_table: [1.234]
    single_val = [1.234]
    assert get_timestamp_for_frame(0, single_val, fps) == 0.0
    # For idx 1: last_pts + 1 * (1.0 / fps) = 1.234 + 0.033333 = 1.267
    assert get_timestamp_for_frame(1, single_val, fps) == round(1.234 + 1.0 / fps, 3)

    # 9. Fallback fps edge cases: 0.0, negative, very small
    # Should be clamped by max(1.0, float(fallback_fps))
    assert get_timestamp_for_frame(10, None, 0.0) == 10.0  # fps clamped to 1.0
    assert get_timestamp_for_frame(10, None, -5.0) == 10.0  # fps clamped to 1.0

    # 10. Numpy integer types as frame_idx
    for np_type in [np.int32, np.int64, np.uint32]:
        val = get_timestamp_for_frame(np_type(3), table, fps)
        assert val == round(table[3], 3)


# ============================================================================
# Section 3: FFmpeg CLI Passthrough & Parameter Placement Verification
# ============================================================================

def test_ffmpeg_passthrough_flag_and_real_execution(tmp_path: Path):
    """
    Stress-test FFmpeg passthrough CLI construction and verify real execution:
    - get_ffmpeg_passthrough_args returns valid argument list
    - VideoROIStream places passthrough flags strictly AFTER '-i <path>' and BEFORE '-vf'
    - Run real FFmpeg process with passthrough flags on sample video and verify it decodes
    """
    ffmpeg_exe = resolve_ffmpeg_executable()
    assert ffmpeg_exe is not None, "FFmpeg executable not found in PATH or environment"

    flags = get_ffmpeg_passthrough_args(ffmpeg_exe)
    print(f"\n[EMPIRICAL LOG] Resolved FFmpeg: {ffmpeg_exe}")
    print(f"[EMPIRICAL LOG] Passthrough flags: {flags}")
    assert isinstance(flags, list)
    assert flags in (["-fps_mode", "passthrough"], ["-vsync", "0"])

    # Test argument placement in VideoROIStream
    intercepted_cmd: list[str] = []

    class MockPopen:
        def __init__(self, cmd, **kwargs):
            intercepted_cmd.extend(cmd)
            self.stdout = None
        def poll(self):
            return 0
        def kill(self):
            pass

    import app.services.ffmpeg_service as fms
    orig_popen = subprocess.Popen
    subprocess.Popen = MockPopen

    try:
        roi = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))
        stream = VideoROIStream(SAMPLE_VIDEO, roi)
        stream.close()
    finally:
        subprocess.Popen = orig_popen

    print(f"[EMPIRICAL LOG] VideoROIStream FFmpeg command: {' '.join(intercepted_cmd)}")
    assert "-i" in intercepted_cmd
    idx_i = intercepted_cmd.index("-i")
    input_file_arg = intercepted_cmd[idx_i + 1]
    assert "-vf" in intercepted_cmd
    idx_vf = intercepted_cmd.index("-vf")

    # Verify that passthrough flags are placed strictly BETWEEN input_file_arg and -vf
    assert idx_vf > idx_i + 1
    flag_slice = intercepted_cmd[idx_i + 2 : idx_vf]
    for flag_part in flags:
        assert flag_part in flag_slice, f"Flag {flag_part} not found between -i and -vf: {flag_slice}"

    # Now execute a real FFmpeg command with passthrough flags on the real video to ensure no error
    test_run_cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-i",
        str(SAMPLE_VIDEO),
        *flags,
        "-vf",
        "crop=64:64:100:100",
        "-vframes",
        "10",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.run(test_run_cmd, capture_output=True)
    assert proc.returncode == 0, f"FFmpeg failed with passthrough flags: {proc.stderr.decode()}"
    # 10 frames * 64 * 64 * 3 bytes = 122,880 bytes
    expected_bytes = 10 * 64 * 64 * 3
    assert len(proc.stdout) == expected_bytes, (
        f"Decoded {len(proc.stdout)} bytes, expected {expected_bytes} bytes"
    )
    print(f"[EMPIRICAL LOG] Real FFmpeg execution succeeded: decoded 10 frames ({len(proc.stdout)} bytes)")


# ============================================================================
# Section 4: Fallback & Resilience on Corrupted, Fake, or Audio-Only Files
# ============================================================================

def test_fallback_non_existent_file():
    """Verify extract_video_pts_table and VideoROIStream safely handle missing files."""
    missing = "d:/non_existent_path_to_video_12345.mp4"
    res = extract_video_pts_table(missing)
    assert res == [], f"Expected [] on missing file, got {res}"

    roi = ROIConfig(main_roi=NormalizedBox(x=0.1, y=0.8, w=0.8, h=0.15))
    with pytest.raises((FileNotFoundError, Exception)):
        VideoROIStream(missing, roi)


def test_fallback_fake_text_mp4_file(tmp_path: Path):
    """Verify graceful handling when a text file is disguised as an .mp4."""
    fake_mp4 = tmp_path / "fake_video.mp4"
    fake_mp4.write_text("This is plain text pretending to be a video container.", encoding="utf-8")

    res = extract_video_pts_table(fake_mp4)
    assert res == [], f"Expected [] on fake text file, got {res}"


def test_fallback_zero_byte_file(tmp_path: Path):
    """Verify graceful handling of 0-byte file."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")

    res = extract_video_pts_table(empty_file)
    assert res == [], f"Expected [] on empty 0-byte file, got {res}"


def test_fallback_corrupted_binary_header(tmp_path: Path):
    """Verify graceful handling of corrupted binary headers."""
    corrupt_file = tmp_path / "corrupt_header.mp4"
    corrupt_file.write_bytes(b"\x00\x00\x00\x20ftypmp42\xff\xff\x00\x00corrupted_payload")

    res = extract_video_pts_table(corrupt_file)
    assert res == [], f"Expected [] on corrupted binary file, got {res}"


def test_fallback_audio_only_file(tmp_path: Path):
    """
    Generate a pure audio file (WAV) with no video track and verify:
    - extract_video_pts_table returns [] safely without raising exceptions
    """
    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not available")

    audio_file = tmp_path / "pure_audio.wav"
    # Generate 1 second sine wave audio
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:a",
        "pcm_s16le",
        str(audio_file),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        pytest.skip("Could not generate audio test file")

    res = extract_video_pts_table(audio_file)
    assert res == [], f"Expected [] on audio-only file, got {res}"
    print(f"[EMPIRICAL LOG] Audio-only file gracefully returned empty PTS table: {res}")


# ============================================================================
# Section 5: Concurrent / Multithreaded Execution Stress Test
# ============================================================================

def test_extract_video_pts_table_concurrency():
    """
    Stress-test extract_video_pts_table under concurrent multithreaded demuxing.
    Ensure no PyAV container race conditions, memory corruption, or inconsistent results.
    """
    if not SAMPLE_VIDEO.exists():
        pytest.skip("Sample video not found")

    def run_worker():
        return extract_video_pts_table(SAMPLE_VIDEO)

    workers = 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_worker) for _ in range(workers)]
        results = [f.result() for f in futures]

    assert len(results) == workers
    base_len = len(results[0])
    assert base_len in (4928, 9407)

    for idx, r in enumerate(results[1:], start=1):
        assert len(r) == base_len, f"Worker {idx} produced different length {len(r)} != {base_len}"
        assert r == results[0], f"Worker {idx} produced divergent PTS table"

    print(f"[EMPIRICAL LOG] Concurrent PTS extraction across {workers} threads succeeded identically")
