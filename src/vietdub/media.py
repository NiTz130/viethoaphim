from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def build_extract_audio_command(video: Path, output: Path, sample_rate: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output),
    ]


def extract_audio(video: Path, output: Path, sample_rate: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_extract_audio_command(video, output, sample_rate))


def probe_media(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def build_mux_preview_command(video: Path, audio: Path, subtitles: Path, output: Path) -> list[str]:
    escaped_subtitles = str(subtitles).replace("\\", "/").replace(":", "\\:")
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        f"subtitles='{escaped_subtitles}'",
        "-shortest",
        str(output),
    ]


def mux_preview(video: Path, audio: Path, subtitles: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_mux_preview_command(video, audio, subtitles, output))
