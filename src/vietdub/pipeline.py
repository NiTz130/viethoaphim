from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from .context import build_context_bundle
from .jobs import Job, JobManager
from .merge import merge_segments
from .models import StepName, TimedSegment, TranslationRow
from .ocr import FixtureOcrEngine
from .stt import FixtureSttEngine
from .translate import export_review_csv


SAFE_SEGMENT_ID_RE = re.compile(r"^m-\d+$")


def write_segments(job: Job, relative: str, segments: list[TimedSegment]) -> None:
    job.write_json(relative, [segment.model_dump() for segment in segments])


def run_fixture_pipeline(video: Path, jobs_dir: Path, stt_fixture: Path, ocr_fixture: Path) -> Job:
    job = JobManager(jobs_dir).create(video, series=None)
    stt_segments = FixtureSttEngine(stt_fixture).transcribe(job.root / "audio" / "original.wav")
    ocr_segments = FixtureOcrEngine(ocr_fixture).recognize(job.input_video)
    merged = merge_segments(stt_segments, ocr_segments)
    context_bundle = build_context_bundle(merged, series_context={})
    translations = [
        TranslationRow(
            segment_id=segment.id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            speaker=segment.speaker,
            text_cn=segment.text,
            text_vi="",
        )
        for segment in merged
    ]

    write_segments(job, "stt/segments.json", stt_segments)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    return job


def run_review_pipeline(video: Path, jobs_dir: Path, series: str | None, settings, review: bool = True) -> Job:
    from .config import Settings
    from .media import extract_audio
    from .memory import collect_system_memory, select_relevant_memory
    from .ocr import PaddleSubtitleOcrEngine
    from .reference import build_reference_context
    from .stt import FasterWhisperSttEngine
    from .translate import translate_with_llm

    typed_settings: Settings = settings
    job = JobManager(jobs_dir).create(video, series=series)
    audio_path = job.root / "audio" / "original.wav"
    extract_audio(job.input_video, audio_path, typed_settings.sample_rate)
    job.mark_done(StepName.EXTRACT, {"audio": str(audio_path)})

    stt_segments = FasterWhisperSttEngine(typed_settings.stt_model, typed_settings.stt_language).transcribe(audio_path)
    write_segments(job, "stt/segments.json", stt_segments)
    job.mark_done(StepName.STT, {"segments": len(stt_segments), "path": str(job.root / "stt" / "segments.json")})

    ocr_segments = PaddleSubtitleOcrEngine().recognize(job.input_video)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    job.mark_done(StepName.OCR, {"segments": len(ocr_segments), "path": str(job.root / "ocr" / "subtitles.json")})

    merged = merge_segments(stt_segments, ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    job.mark_done(StepName.MERGE, {"segments": len(merged), "path": str(job.root / "transcript" / "merged.json")})

    reference_context = build_reference_context(merged, Path(typed_settings.reference_data_dir))
    raw_system_memory = collect_system_memory(jobs_dir, current_job=job.root)
    selected_system_memory = select_relevant_memory(raw_system_memory, merged)
    context_bundle = build_context_bundle(
        merged,
        series_context={},
        reference_context=reference_context,
        system_memory=selected_system_memory,
    )
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    if raw_system_memory.warnings:
        job.write_json("context/system_memory_warnings.json", [warning.model_dump() for warning in raw_system_memory.warnings])
    job.mark_done(
        StepName.CONTEXT,
        {
            "context_files": len(context_bundle),
            "warnings": len(raw_system_memory.warnings),
        },
    )

    translations = translate_with_llm(
        merged,
        context_bundle,
        settings=typed_settings,
        review=review,
    )
    job.write_json("translation/translated.json", [row.model_dump() for row in translations])
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    job.mark_done(
        StepName.TRANSLATE,
        {
            "rows": len(translations),
            "review_csv": str(job.root / "translation" / "review.csv"),
        },
    )
    return job


def _load_allowed_segment_ids(job: Job) -> set[str] | None:
    merged_path = job.root / "transcript" / "merged.json"
    if not merged_path.exists():
        return None
    try:
        data = json.loads(merged_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid transcript file {merged_path}: invalid JSON") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid transcript file {merged_path}: expected a JSON list")

    allowed: set[str] = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise RuntimeError(f"Invalid transcript file {merged_path}: item {index} is missing string id")
        allowed.add(item["id"])
    return allowed


def _is_safe_segment_path_component(segment_id: str) -> bool:
    if not segment_id:
        return False
    if "/" in segment_id or "\\" in segment_id or ":" in segment_id:
        return False
    return True


def _is_safe_fallback_segment_id(segment_id: str) -> bool:
    if not _is_safe_segment_path_component(segment_id):
        return False
    return SAFE_SEGMENT_ID_RE.fullmatch(segment_id) is not None


def _validate_resume_segment_ids(job: Job, rows: list[TranslationRow]) -> None:
    allowed_ids = _load_allowed_segment_ids(job)
    review_path = job.root / "translation" / "review.csv"
    active_ids: set[str] = set()
    for row in rows:
        if row.status == "skip" or not row.text_vi.strip():
            continue
        segment_id = row.segment_id
        if row.start_ms < 0 or row.end_ms <= row.start_ms:
            raise RuntimeError(
                f"Invalid timing in {review_path}: {segment_id!r} start_ms={row.start_ms} end_ms={row.end_ms}"
            )
        if not _is_safe_segment_path_component(segment_id):
            raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
        if allowed_ids is not None:
            if segment_id not in allowed_ids:
                raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r} is not in transcript/merged.json")
        elif not _is_safe_fallback_segment_id(segment_id):
            raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
        if segment_id in active_ids:
            raise RuntimeError(f"Duplicate segment_id in {review_path}: {segment_id!r}")
        active_ids.add(segment_id)


def _probe_video_duration_ms(job: Job) -> int | None:
    from .media import probe_media

    try:
        metadata = probe_media(job.input_video)
    except (RuntimeError, OSError, ValueError):
        return None
    format_metadata = metadata.get("format") if isinstance(metadata, dict) else None
    if not isinstance(format_metadata, dict):
        return None
    duration = format_metadata.get("duration")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return None
    return int(duration_seconds * 1000)


def resume_tts_and_render(job: Job, settings) -> Path:
    """Re-run TTS + render from a finalized ``review.csv``.

    Validation runs before any stale render artifacts are removed, so a
    bad edit in ``review.csv`` leaves the previous render in place for the
    user to recover instead of silently destroying it.
    """
    import asyncio

    from .media import mux_preview
    from .srt import render_srt
    from .sync import assemble_final_audio
    from .translate import import_review_csv
    from .tts import MiniMaxTtsEngine

    srt_path = job.root / "output" / "subtitles_vi.srt"
    final_audio = job.root / "tts" / "final_vi.wav"
    sync_report_path = job.root / "tts" / "sync_report.json"
    preview = job.root / "output" / "preview_vi.mp4"

    rows = import_review_csv(job.root / "translation" / "review.csv")
    _validate_resume_segment_ids(job, rows)

    status = job.load_status()
    if StepName.RENDER.value in status:
        del status[StepName.RENDER.value]
        job.write_json("status.json", status)
    srt_path.unlink(missing_ok=True)
    final_audio.unlink(missing_ok=True)
    sync_report_path.unlink(missing_ok=True)
    preview.unlink(missing_ok=True)
    vietnamese_segments = [
        TimedSegment(
            id=row.segment_id,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            text=row.text_vi,
            speaker=row.speaker,
            source="translation",
        )
        for row in rows
        if row.status != "skip" and row.text_vi.strip()
    ]
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(render_srt(vietnamese_segments), encoding="utf-8")

    async def synthesize_all() -> int:
        engine = MiniMaxTtsEngine(
            voice_id=settings.tts_voice_id,
            api_key=settings.tts_api_key or settings.anthropic_api_key,
            base_url=settings.tts_base_url,
            model=settings.tts_model,
        )
        warnings: list[dict[str, str]] = []
        warning_path = job.root / "tts" / "tts_warnings.json"
        for row in rows:
            if row.status == "skip" or not row.text_vi.strip():
                continue
            segment_audio = job.root / "tts" / "segments" / f"{row.segment_id}.mp3"
            segment_audio.unlink(missing_ok=True)
            try:
                await engine.synthesize_segment(row, segment_audio)
            except Exception as exc:  # noqa: BLE001 - best-effort synthesis: any single-segment TTS failure must not abort the pipeline.
                segment_audio.unlink(missing_ok=True)
                warnings.append({"segment_id": row.segment_id, "error": str(exc)})
        if warnings:
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.write_text(json.dumps(warnings, ensure_ascii=False, indent=2), encoding="utf-8")
        elif warning_path.exists():
            warning_path.unlink()
        return len(warnings)

    warning_count = asyncio.run(synthesize_all())
    job.mark_done(StepName.TTS, {"segments": len(vietnamese_segments), "warnings": warning_count})

    try:
        sync_report = assemble_final_audio(
            rows=rows,
            segment_dir=job.root / "tts" / "segments",
            output_audio=final_audio,
            sample_rate=settings.sample_rate,
            video_duration_ms=_probe_video_duration_ms(job),
            sync_report_path=sync_report_path,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):  # do not leave stale render artifacts after a failed resume.
        final_audio.unlink(missing_ok=True)
        preview.unlink(missing_ok=True)
        raise
    if not final_audio.exists() or final_audio.stat().st_size == 0:
        final_audio.unlink(missing_ok=True)
        preview.unlink(missing_ok=True)
        raise RuntimeError(f"Final audio was not created: {final_audio}")

    preview.unlink(missing_ok=True)
    try:
        mux_preview(job.input_video, final_audio, srt_path, preview)
    except (OSError, RuntimeError, subprocess.SubprocessError):  # do not leave stale or partial previews after mux failure.
        preview.unlink(missing_ok=True)
        raise
    if not preview.exists() or preview.stat().st_size == 0:
        preview.unlink(missing_ok=True)
        raise RuntimeError(f"Preview was not created: {preview}")
    synced_segments = sum(1 for segment in sync_report.segments if segment.synced_duration_ms > 0)
    skipped_segments = sum(1 for segment in sync_report.segments if segment.synced_duration_ms == 0)
    job.mark_done(
        StepName.RENDER,
        {
            "subtitles": str(srt_path),
            "segments": len(vietnamese_segments),
            "final_audio": str(final_audio),
            "sync_report": str(sync_report_path),
            "preview": str(preview),
            "synced_segments": synced_segments,
            "skipped_segments": skipped_segments,
        },
    )
    return srt_path
