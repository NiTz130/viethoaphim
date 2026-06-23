from __future__ import annotations

import subprocess
import wave
from pathlib import Path


def _format_atempo_value(value: float) -> str:
    text = f"{value:.3g}"
    if "." not in text and "e" not in text:
        text = f"{text}.0"
    return text


def speed_factor_for_duration(actual_ms: int, target_ms: int) -> float:
    if actual_ms <= 0 or target_ms <= 0:
        raise ValueError("durations must be positive")
    return round(actual_ms / target_ms, 3)


def atempo_filters(factor: float) -> list[str]:
    filters: list[str] = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={_format_atempo_value(remaining)}")
    return filters


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return int(frames / rate * 1000)


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def trim_silence(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=1:stop_threshold=-45dB",
            str(output_path),
        ]
    )


def stretch_audio(input_path: Path, output_path: Path, factor: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = ",".join(atempo_filters(factor))
    run_ffmpeg(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", filters, str(output_path)])
