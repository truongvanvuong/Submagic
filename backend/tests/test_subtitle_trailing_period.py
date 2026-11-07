"""Unit tests for subtitle trailing period stripping logic."""
import pytest

from app.schemas import BilingualLayout, ExportFormat, SubtitleSegment
from app.utils.subtitle_formatter import (
    format_subtitles,
    remove_trailing_period,
    remove_trailing_period_from_line,
)


def test_remove_trailing_period_standard_sentences():
    assert remove_trailing_period("Xin chào thế giới.") == "Xin chào thế giới"
    assert remove_trailing_period("Hello world.") == "Hello world"
    assert remove_trailing_period("Tôi đang học tiếng Anh.") == "Tôi đang học tiếng Anh"


def test_remove_trailing_period_preserves_question_and_exclamation():
    assert remove_trailing_period("Bạn khỏe không?") == "Bạn khỏe không?"
    assert remove_trailing_period("Tuyệt vời quá!") == "Tuyệt vời quá!"
    assert remove_trailing_period("Thật không?!") == "Thật không?!"


def test_remove_trailing_period_preserves_ellipsis():
    assert remove_trailing_period("Đang suy nghĩ...") == "Đang suy nghĩ..."
    assert remove_trailing_period("Đang tải…") == "Đang tải…"
    assert remove_trailing_period("Chờ một lát....") == "Chờ một lát...."


def test_remove_trailing_period_preserves_middle_periods():
    assert remove_trailing_period("Câu một. Câu hai") == "Câu một. Câu hai"
    assert remove_trailing_period("Câu một. Câu hai.") == "Câu một. Câu hai"
    assert remove_trailing_period("Hôm nay trời nắng. Mai có thể mưa.") == "Hôm nay trời nắng. Mai có thể mưa"


def test_remove_trailing_period_preserves_decimals():
    assert remove_trailing_period("Điểm số là 3.5") == "Điểm số là 3.5"
    assert remove_trailing_period("Tỉ lệ 3.5.") == "Tỉ lệ 3.5"
    assert remove_trailing_period("Phiên bản 2.0.1.") == "Phiên bản 2.0.1"


def test_remove_trailing_period_preserves_abbreviations():
    assert remove_trailing_period("Chào Mr. Brown.") == "Chào Mr. Brown"
    assert remove_trailing_period("Tôi đã gặp Mr.") == "Tôi đã gặp Mr."
    assert remove_trailing_period("Liên hệ Dr.") == "Liên hệ Dr."
    assert remove_trailing_period("Sách, vở, bút, etc.") == "Sách, vở, bút, etc."
    assert remove_trailing_period("Tác giả J. K.") == "Tác giả J. K."


def test_remove_trailing_period_multiline():
    multiline = "Dòng thứ nhất.\nDòng thứ hai."
    expected = "Dòng thứ nhất\nDòng thứ hai"
    assert remove_trailing_period(multiline) == expected

    mixed = "Dòng một có Mr. Bean.\nDòng hai: Bạn có ổn không?\nDòng ba kết thúc..."
    expected_mixed = "Dòng một có Mr. Bean\nDòng hai: Bạn có ổn không?\nDòng ba kết thúc..."
    assert remove_trailing_period(mixed) == expected_mixed


def test_format_subtitles_applies_to_all_formats():
    segments = [
        SubtitleSegment(
            project_id="test_proj",
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="Hôm nay là một ngày tuyệt vời.",
            translated_text="Today is a wonderful day.",
        ),
        SubtitleSegment(
            project_id="test_proj",
            index=2,
            start_time=2.0,
            end_time=4.0,
            text="Bạn có nghĩ vậy không?",
            translated_text="Do you think so?",
        ),
        SubtitleSegment(
            project_id="test_proj",
            index=3,
            start_time=4.0,
            end_time=6.0,
            text="Tỉ lệ thành công là 9.5.",
            translated_text="Success rate is 9.5.",
        ),
    ]

    # SRT format
    srt = format_subtitles(segments, ExportFormat.SRT, BilingualLayout.ORIGINAL_TOP)
    assert "Hôm nay là một ngày tuyệt vời\nToday is a wonderful day" in srt
    assert "Bạn có nghĩ vậy không?\nDo you think so?" in srt
    assert "Tỉ lệ thành công là 9.5\nSuccess rate is 9.5" in srt

    # VTT format
    vtt = format_subtitles(segments, ExportFormat.VTT, BilingualLayout.TRANSLATED_ONLY)
    assert "Today is a wonderful day" in vtt
    assert "Today is a wonderful day." not in vtt
    assert "Do you think so?" in vtt
    assert "Success rate is 9.5" in vtt

    # TXT format without timestamps
    txt = format_subtitles(segments, ExportFormat.TXT, BilingualLayout.NONE, include_timestamps_in_txt=False)
    lines = txt.strip().split("\n")
    assert lines[0] == "Hôm nay là một ngày tuyệt vời"
    assert lines[1] == "Bạn có nghĩ vậy không?"
    assert lines[2] == "Tỉ lệ thành công là 9.5"
