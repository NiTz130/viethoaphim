import json

import pytest
from pydub.generators import Sine

from vietdub.models import TranslationRow
from vietdub.sync import assemble_final_audio, atempo_filters, speed_factor_for_duration, wav_duration_ms


def _row(segment_id: str, start_ms: int, end_ms: int, text_vi: str = "Xin chao", status: str = "reviewed") -> TranslationRow:
    return TranslationRow(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=None,
        text_cn="source",
        text_vi=text_vi,
        context_note="",
        status=status,
    )


def _write_mp3(path, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sine(440).to_audio_segment(duration=duration_ms).export(path, format="mp3")


def test_speed_factor_for_duration():
    assert speed_factor_for_duration(actual_ms=2000, target_ms=1000) == 2.0
    assert speed_factor_for_duration(actual_ms=1000, target_ms=2000) == 0.5


def test_atempo_filters_split_large_factor():
    assert atempo_filters(4.0) == ["atempo=2.0", "atempo=2.0"]
    assert atempo_filters(0.25) == ["atempo=0.5", "atempo=0.5"]


def test_assemble_final_audio_creates_timeline_and_report(tmp_path):
    segment_dir = tmp_path / "segments"
    _write_mp3(segment_dir / "m-0001.mp3", 400)
    _write_mp3(segment_dir / "m-0002.mp3", 1600)
    output_audio = tmp_path / "final_vi.wav"
    report_path = tmp_path / "sync_report.json"

    report = assemble_final_audio(
        rows=[_row("m-0001", 0, 1000), _row("m-0002", 1500, 2500)],
        segment_dir=segment_dir,
        output_audio=output_audio,
        sample_rate=44_100,
        video_duration_ms=3000,
        sync_report_path=report_path,
    )

    assert output_audio.exists()
    assert 2950 <= wav_duration_ms(output_audio) <= 3050
    assert report.output_audio == output_audio
    assert [segment.segment_id for segment in report.segments] == ["m-0001", "m-0002"]
    assert report.segments[0].original_duration_ms > 0
    assert report.segments[0].synced_duration_ms == 1000
    assert report.segments[1].synced_duration_ms == 1000
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["output_audio"] == str(output_audio)
    assert saved["segments"][0]["segment_id"] == "m-0001"


def test_assemble_final_audio_records_missing_and_zero_byte_warnings(tmp_path):
    segment_dir = tmp_path / "segments"
    _write_mp3(segment_dir / "m-0001.mp3", 500)
    zero_path = segment_dir / "m-0003.mp3"
    zero_path.parent.mkdir(parents=True, exist_ok=True)
    zero_path.write_bytes(b"")
    output_audio = tmp_path / "final_vi.wav"
    report_path = tmp_path / "sync_report.json"

    report = assemble_final_audio(
        rows=[
            _row("m-0001", 0, 1000),
            _row("m-0002", 1000, 2000),
            _row("m-0003", 2000, 3000),
        ],
        segment_dir=segment_dir,
        output_audio=output_audio,
        sample_rate=44_100,
        video_duration_ms=None,
        sync_report_path=report_path,
    )

    warnings_by_id = {segment.segment_id: segment.warnings for segment in report.segments}
    assert output_audio.exists()
    assert warnings_by_id["m-0001"] == []
    assert "missing TTS audio" in warnings_by_id["m-0002"][0]
    assert "empty TTS audio" in warnings_by_id["m-0003"][0]
    assert json.loads(report_path.read_text(encoding="utf-8"))["segments"][2]["synced_duration_ms"] == 0


def test_assemble_final_audio_rejects_unsafe_segment_id_without_reading_outside_segment_dir(tmp_path):
    segment_dir = tmp_path / "segments"
    _write_mp3(segment_dir / "m-0001.mp3", 500)
    _write_mp3(tmp_path / "evil.mp3", 500)
    output_audio = tmp_path / "final_vi.wav"

    report = assemble_final_audio(
        rows=[_row("..\\evil", 0, 1000), _row("m-0001", 1000, 2000)],
        segment_dir=segment_dir,
        output_audio=output_audio,
        sample_rate=44_100,
        video_duration_ms=None,
    )

    warnings_by_id = {segment.segment_id: segment.warnings for segment in report.segments}
    assert output_audio.exists()
    assert "invalid segment_id" in warnings_by_id["..\\evil"][0]
    assert report.segments[0].synced_duration_ms == 0
    assert warnings_by_id["m-0001"] == []
    assert report.segments[1].synced_duration_ms == 1000


def test_assemble_final_audio_records_invalid_duration_warning(tmp_path):
    segment_dir = tmp_path / "segments"
    _write_mp3(segment_dir / "m-0001.mp3", 500)
    _write_mp3(segment_dir / "m-0002.mp3", 500)
    output_audio = tmp_path / "final_vi.wav"

    report = assemble_final_audio(
        rows=[_row("m-0001", 1000, 1000), _row("m-0002", 1000, 2000)],
        segment_dir=segment_dir,
        output_audio=output_audio,
        sample_rate=44_100,
        video_duration_ms=None,
    )

    warnings_by_id = {segment.segment_id: segment.warnings for segment in report.segments}
    assert output_audio.exists()
    assert "invalid subtitle duration" in warnings_by_id["m-0001"][0]
    assert report.segments[0].synced_duration_ms == 0
    assert warnings_by_id["m-0002"] == []


def test_assemble_final_audio_records_unreadable_non_empty_audio_warning(tmp_path):
    segment_dir = tmp_path / "segments"
    _write_mp3(segment_dir / "m-0001.mp3", 500)
    bad_audio_path = segment_dir / "m-0002.mp3"
    bad_audio_path.write_text("not an mp3", encoding="utf-8")
    output_audio = tmp_path / "final_vi.wav"

    report = assemble_final_audio(
        rows=[_row("m-0001", 0, 1000), _row("m-0002", 1000, 2000)],
        segment_dir=segment_dir,
        output_audio=output_audio,
        sample_rate=44_100,
        video_duration_ms=None,
    )

    warnings_by_id = {segment.segment_id: segment.warnings for segment in report.segments}
    assert output_audio.exists()
    assert warnings_by_id["m-0001"] == []
    assert "unreadable TTS audio" in warnings_by_id["m-0002"][0]
    assert report.segments[1].synced_duration_ms == 0


def test_assemble_final_audio_fails_when_no_valid_audio(tmp_path):
    report_path = tmp_path / "sync_report.json"
    output_audio = tmp_path / "final_vi.wav"

    with pytest.raises(RuntimeError, match="No valid TTS segment audio"):
        assemble_final_audio(
            rows=[_row("m-0001", 0, 1000)],
            segment_dir=tmp_path / "segments",
            output_audio=output_audio,
            sample_rate=44_100,
            video_duration_ms=1000,
            sync_report_path=report_path,
        )

    assert not output_audio.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["segments"][0]["segment_id"] == "m-0001"
    assert "missing TTS audio" in saved["segments"][0]["warnings"][0]
