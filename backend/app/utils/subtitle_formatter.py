"""Subtitle formatting and export engine for .SRT, .VTT, and .TXT (including Bilingual support)."""
from __future__ import annotations

import textwrap

from app.schemas import BilingualLayout, ExportFormat, SubtitleSegment


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_timestamp(timestamp_str: str) -> float:
    """Parse an SRT or WebVTT timestamp into seconds as a float.

    Supports:
      - SRT format: 'HH:MM:SS,mmm'
      - WebVTT format: 'HH:MM:SS.mmm' or 'MM:SS.mmm'
      - Single seconds format: 'SS.mmm'
    """
    ts = timestamp_str.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return round(hours * 3600.0 + minutes * 60.0 + seconds, 3)
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return round(minutes * 60.0 + seconds, 3)
    elif len(parts) == 1:
        return round(float(parts[0]), 3)
    else:
        raise ValueError(f"Invalid timestamp format: '{timestamp_str}'")


def parse_timestamp_srt(timestamp_str: str) -> float:
    """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
    return parse_timestamp(timestamp_str)


def parse_timestamp_vtt(timestamp_str: str) -> float:
    """Parse WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    return parse_timestamp(timestamp_str)


def clean_subtitle_text(text: str) -> str:
    """Strip whitespace, collapse repeated blank lines, and normalize newlines."""
    if not text:
        return ""
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank and cleaned_lines:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def wrap_subtitle_lines(text: str, max_chars: int = 42) -> str:
    """Wrap lines in subtitle text if any single line exceeds max_chars, keeping words intact."""
    if not text:
        return ""
    paragraphs = text.split("\n")
    wrapped_lines: list[str] = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            wrapped_lines.append("")
            continue
        if len(p_clean) <= max_chars:
            wrapped_lines.append(p_clean)
        else:
            wrapped = textwrap.wrap(p_clean, width=max_chars, break_long_words=False)
            wrapped_lines.extend(wrapped if wrapped else [p_clean])
    return "\n".join(wrapped_lines)


def _resolve_segment_text(
    seg: SubtitleSegment,
    layout: BilingualLayout,
    wrap_lines: bool = False,
    max_chars: int = 42,
) -> str:
    orig = clean_subtitle_text(seg.text or "")
    trans = clean_subtitle_text(seg.translated_text or "")

    if layout == BilingualLayout.TRANSLATED_ONLY:
        resolved = trans if trans else orig
    elif layout == BilingualLayout.ORIGINAL_TOP:
        if orig and trans:
            resolved = f"{orig}\n{trans}"
        else:
            resolved = orig or trans
    elif layout == BilingualLayout.TRANSLATED_TOP:
        if orig and trans:
            resolved = f"{trans}\n{orig}"
        else:
            resolved = trans or orig
    else:  # BilingualLayout.NONE
        resolved = orig

    if wrap_lines and resolved:
        resolved = wrap_subtitle_lines(resolved, max_chars=max_chars)
    return resolved


def format_subtitles(
    segments: list[SubtitleSegment],
    export_format: ExportFormat = ExportFormat.SRT,
    bilingual_layout: BilingualLayout = BilingualLayout.NONE,
    include_timestamps_in_txt: bool = False,
    wrap_lines: bool = False,
    max_chars_per_line: int = 42,
) -> str:
    """Render subtitle segments into .srt, .vtt, or .txt string."""
    sorted_segs = sorted(segments, key=lambda s: (s.start_time, s.index))

    if export_format == ExportFormat.SRT:
        blocks: list[str] = []
        for idx, seg in enumerate(sorted_segs, start=1):
            body = _resolve_segment_text(
                seg, bilingual_layout, wrap_lines=wrap_lines, max_chars=max_chars_per_line
            )
            if not body:
                continue
            start_str = format_timestamp_srt(seg.start_time)
            end_str = format_timestamp_srt(max(seg.start_time + 0.1, seg.end_time))
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{body}")
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    if export_format == ExportFormat.VTT:
        blocks = ["WEBVTT", ""]
        for idx, seg in enumerate(sorted_segs, start=1):
            body = _resolve_segment_text(
                seg, bilingual_layout, wrap_lines=wrap_lines, max_chars=max_chars_per_line
            )
            if not body:
                continue
            start_str = format_timestamp_vtt(seg.start_time)
            end_str = format_timestamp_vtt(max(seg.start_time + 0.1, seg.end_time))
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{body}\n")
        return "\n".join(blocks)

    # ExportFormat.TXT
    lines: list[str] = []
    for seg in sorted_segs:
        body = _resolve_segment_text(
            seg, bilingual_layout, wrap_lines=wrap_lines, max_chars=max_chars_per_line
        )
        if not body:
            continue
        if include_timestamps_in_txt:
            start_str = format_timestamp_vtt(seg.start_time)
            end_str = format_timestamp_vtt(seg.end_time)
            single_line = body.replace("\n", " | ")
            lines.append(f"[{start_str} -> {end_str}] {single_line}")
        else:
            lines.append(body.replace("\n", " "))
    return "\n".join(lines) + ("\n" if lines else "")
