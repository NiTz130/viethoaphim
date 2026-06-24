from __future__ import annotations

import json
from pathlib import Path

from .context import build_context_bundle
from .jobs import Job, JobManager
from .merge import merge_segments
from .models import StepName, TimedSegment, TranslationRow
from .ocr import FixtureOcrEngine
from .stt import FixtureSttEngine
from .translate import export_review_csv


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


def run_review_pipeline(video: Path, jobs_dir: Path, series: str | None, settings) -> Job:
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
    ocr_segments = PaddleSubtitleOcrEngine().recognize(job.input_video)
    merged = merge_segments(stt_segments, ocr_segments)
    reference_context = build_reference_context(merged, Path(typed_settings.reference_data_dir))
    raw_system_memory = collect_system_memory(jobs_dir, current_job=job.root)
    selected_system_memory = select_relevant_memory(raw_system_memory, merged)
    context_bundle = build_context_bundle(
        merged,
        series_context={},
        reference_context=reference_context,
        system_memory=selected_system_memory,
    )
    translations = translate_with_llm(
        merged,
        context_bundle,
        typed_settings.openai_api_key,
        typed_settings.llm_model,
        typed_settings.openai_base_url,
    )

    write_segments(job, "stt/segments.json", stt_segments)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    if raw_system_memory.warnings:
        job.write_json("context/system_memory_warnings.json", [warning.model_dump() for warning in raw_system_memory.warnings])
    job.write_json("translation/translated.json", [row.model_dump() for row in translations])
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    return job


def resume_tts_and_render(job: Job, settings) -> Path:
    import asyncio

    from .media import mux_preview
    from .srt import render_srt
    from .translate import import_review_csv
    from .tts import EdgeTtsEngine

    rows = import_review_csv(job.root / "translation" / "review.csv")
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
    srt_path = job.root / "output" / "subtitles_vi.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(render_srt(vietnamese_segments), encoding="utf-8")

    async def synthesize_all() -> None:
        engine = EdgeTtsEngine(settings.edge_voice)
        warnings: list[dict[str, str]] = []
        for row in rows:
            if row.status == "skip" or not row.text_vi.strip():
                continue
            try:
                await engine.synthesize_segment(row, job.root / "tts" / "segments" / f"{row.segment_id}.mp3")
            except Exception as exc:  # noqa: BLE001 - keep subtitle output even if one TTS request fails.
                warnings.append({"segment_id": row.segment_id, "error": str(exc)})
        if warnings:
            warning_path = job.root / "tts" / "tts_warnings.json"
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.write_text(json.dumps(warnings, ensure_ascii=False, indent=2), encoding="utf-8")

    asyncio.run(synthesize_all())
    final_audio = job.root / "tts" / "final_vi.wav"
    final_audio.write_bytes(b"")
    preview = job.root / "output" / "preview_vi.mp4"
    if final_audio.stat().st_size > 0:
        mux_preview(job.input_video, final_audio, srt_path, preview)
    return srt_path
