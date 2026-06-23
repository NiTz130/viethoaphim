from __future__ import annotations

from pathlib import Path

from .context import build_context_bundle
from .jobs import Job, JobManager
from .merge import merge_segments
from .models import TimedSegment, TranslationRow
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
