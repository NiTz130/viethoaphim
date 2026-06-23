from pathlib import Path

from vietdub.media import build_extract_audio_command, build_mux_preview_command
from vietdub.models import TimedSegment
from vietdub.srt import render_srt


def test_render_srt():
    text = render_srt([TimedSegment(id="1", start_ms=0, end_ms=1500, text="Xin ch\u00e0o")])
    assert "1\n00:00:00,000 --> 00:00:01,500\nXin ch\u00e0o\n" in text


def test_extract_audio_command_uses_wav_and_sample_rate(tmp_path):
    command = build_extract_audio_command(tmp_path / "in.mp4", tmp_path / "out.wav", 44100)
    assert command[:2] == ["ffmpeg", "-y"]
    assert "-ar" in command
    assert "44100" in command


def test_mux_preview_command_maps_video_audio_and_subtitles(tmp_path):
    command = build_mux_preview_command(
        video=tmp_path / "in.mp4",
        audio=tmp_path / "vi.wav",
        subtitles=tmp_path / "vi.srt",
        output=tmp_path / "preview.mp4",
    )
    joined = " ".join(str(part) for part in command)
    assert "subtitles=" in joined
    assert str(tmp_path / "preview.mp4") in joined
