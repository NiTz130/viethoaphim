from __future__ import annotations

import json
import subprocess
import wave
from collections.abc import Iterable
from pathlib import Path

from pydub import AudioSegment

from .models import SyncReport, SyncSegmentReport, TranslationRow


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


def _active_translation_rows(rows: Iterable[TranslationRow]) -> list[TranslationRow]:
    return [row for row in rows if row.status != "skip" and row.text_vi.strip()]


def _timeline_duration_ms(rows: list[TranslationRow], video_duration_ms: int | None) -> int:
    row_duration_ms = max((row.end_ms for row in rows if row.end_ms > 0), default=0)
    if video_duration_ms is not None and video_duration_ms > 0:
        return max(video_duration_ms, row_duration_ms)
    return row_duration_ms


def _fit_audio_to_duration(segment: AudioSegment, target_ms: int, sample_rate: int) -> tuple[AudioSegment, int, float]:
    normalized = segment.set_channels(1).set_frame_rate(sample_rate)
    original_duration_ms = len(normalized)
    factor = speed_factor_for_duration(original_duration_ms, target_ms)
    if factor != 1.0:
        new_frame_rate = max(1, int(normalized.frame_rate * factor))
        normalized = normalized._spawn(normalized.raw_data, overrides={"frame_rate": new_frame_rate})
        normalized = normalized.set_frame_rate(sample_rate)

    if len(normalized) > target_ms:
        normalized = normalized[:target_ms]
    elif len(normalized) < target_ms:
        padding = AudioSegment.silent(duration=target_ms - len(normalized), frame_rate=sample_rate).set_channels(1)
        normalized += padding
    return normalized, original_duration_ms, factor


def _segment_audio_path(segment_dir: Path, segment_id: str) -> Path | None:
    if not segment_id or "/" in segment_id or "\\" in segment_id or ":" in segment_id:
        return None
    return segment_dir / f"{segment_id}.mp3"


def _write_sync_report(report: SyncReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def assemble_final_audio(
    rows: Iterable[TranslationRow],
    segment_dir: Path,
    output_audio: Path,
    sample_rate: int = 44_100,
    video_duration_ms: int | None = None,
    sync_report_path: Path | None = None,
) -> SyncReport:
    active_rows = _active_translation_rows(rows)
    report = SyncReport(output_audio=output_audio, sample_rate=sample_rate)
    timeline_ms = max(1, _timeline_duration_ms(active_rows, video_duration_ms))
    timeline = AudioSegment.silent(duration=timeline_ms, frame_rate=sample_rate).set_channels(1)
    valid_segments = 0

    for row in active_rows:
        warnings: list[str] = []
        target_ms = row.end_ms - row.start_ms
        original_duration_ms = 0
        synced_duration_ms = 0
        speed_factor = 1.0
        segment_path = _segment_audio_path(segment_dir, row.segment_id)

        if segment_path is None:
            warnings.append(f"invalid segment_id for TTS audio: {row.segment_id!r}")
        elif target_ms <= 0:
            warnings.append(f"invalid subtitle duration: start_ms={row.start_ms} end_ms={row.end_ms}")
        elif not segment_path.exists():
            warnings.append(f"missing TTS audio: {segment_path}")
        elif segment_path.stat().st_size == 0:
            warnings.append(f"empty TTS audio: {segment_path}")
        else:
            try:
                raw_segment = AudioSegment.from_file(segment_path)
                fitted_segment, original_duration_ms, speed_factor = _fit_audio_to_duration(raw_segment, target_ms, sample_rate)
                synced_duration_ms = len(fitted_segment)
                timeline = timeline.overlay(fitted_segment, position=max(0, row.start_ms))
                valid_segments += 1
            except Exception as exc:  # noqa: BLE001 - continue with other synthesized segments.
                warnings.append(f"unreadable TTS audio: {exc}")

        report.segments.append(
            SyncSegmentReport(
                segment_id=row.segment_id,
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                original_duration_ms=original_duration_ms,
                synced_duration_ms=synced_duration_ms,
                speed_factor=speed_factor,
                warnings=warnings,
            )
        )

    if valid_segments == 0:
        if sync_report_path is not None:
            _write_sync_report(report, sync_report_path)
        raise RuntimeError("No valid TTS segment audio found; cannot assemble final audio")

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(output_audio, format="wav")
    if sync_report_path is not None:
        _write_sync_report(report, sync_report_path)
    return report
