# Final Audio Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the missing final Vietnamese audio track and muxed preview for `vietdub resume <job> --from tts`.

**Architecture:** Keep Edge TTS isolated in `tts.py`; add final timeline assembly to `sync.py` with `pydub`; let `pipeline.py` orchestrate SRT, synthesis, assembly, sync report, and muxing. The pipeline writes `tts/final_vi.wav`, `tts/sync_report.json`, and `output/preview_vi.mp4`, while preserving per-segment warnings.

**Tech Stack:** Python 3.11, pydantic v2, pydub, FFmpeg/ffprobe, pytest, Typer CLI.

---

## File Structure

- Modify `src/vietdub/sync.py`: add `assemble_final_audio()` and private helpers for active rows, timeline duration, audio fitting, and sync report JSON writing.
- Modify `src/vietdub/pipeline.py`: replace the empty `final_vi.wav` write with real audio assembly, video duration probing, preview muxing, and render status details.
- Modify `tests/test_sync.py`: add pydub-generated audio fixtures and unit tests for final WAV creation, warning collection, and fatal no-valid-audio behavior.
- Modify `tests/test_pipeline.py`: update resume tests to use valid local MP3 fixtures, assert final audio/sync report/preview creation, and assert the all-failed TTS path raises clearly after preserving SRT and warnings.
- Do not modify `pyproject.toml`: `pydub` is already a project dependency and FFmpeg is already required by existing media workflows.

---

### Task 1: Sync Assembly Core

**Files:**
- Modify: `tests/test_sync.py`
- Modify: `src/vietdub/sync.py`

- [ ] **Step 1: Write failing sync tests**

Replace `tests/test_sync.py` with:

```python
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
```

- [ ] **Step 2: Run sync tests to verify failure**

Run:

```powershell
python -m pytest tests/test_sync.py -v
```

Expected: FAIL during collection with `ImportError: cannot import name 'assemble_final_audio'`.

- [ ] **Step 3: Implement final audio assembly**

Replace `src/vietdub/sync.py` with:

```python
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
        segment_path = segment_dir / f"{row.segment_id}.mp3"

        if target_ms <= 0:
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
```

- [ ] **Step 4: Run sync tests to verify pass**

Run:

```powershell
python -m pytest tests/test_sync.py -v
```

Expected: PASS for all tests in `tests/test_sync.py`.

- [ ] **Step 5: Commit sync assembly**

Run:

```powershell
git add tests/test_sync.py src/vietdub/sync.py
git commit -m "feat: assemble final dubbed audio"
```

---

### Task 2: Pipeline Resume Integration

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/vietdub/pipeline.py`

- [ ] **Step 1: Write failing pipeline integration tests**

Add this import near the top of `tests/test_pipeline.py`:

```python
from pydub.generators import Sine
```

Add these helpers below the imports:

```python
def _resume_settings():
    return type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural", "sample_rate": 44_100})()


def _write_valid_mp3(path, duration_ms: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sine(440).to_audio_segment(duration=duration_ms).export(path, format="mp3")


def _patch_resume_media(monkeypatch):
    calls = []

    def fake_probe_media(video_path):
        return {"format": {"duration": "1.000"}, "streams": []}

    def fake_mux_preview(video, audio, subtitles, output):
        calls.append({"video": video, "audio": audio, "subtitles": subtitles, "output": output})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"preview")

    monkeypatch.setattr("vietdub.media.probe_media", fake_probe_media)
    monkeypatch.setattr("vietdub.media.mux_preview", fake_mux_preview)
    return calls
```

Replace `test_resume_tts_and_render_writes_output_srt()` with:

```python
def test_resume_tts_and_render_writes_output_srt(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})
    mux_calls = _patch_resume_media(monkeypatch)

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fake_synthesize_segment)

    srt_path = resume_tts_and_render(job, _resume_settings())

    final_audio = tmp_path / "job" / "tts" / "final_vi.wav"
    sync_report_path = tmp_path / "job" / "tts" / "sync_report.json"
    preview_path = tmp_path / "job" / "output" / "preview_vi.mp4"
    assert srt_path == tmp_path / "job" / "output" / "subtitles_vi.srt"
    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    assert (tmp_path / "job" / "tts" / "segments" / "m-0001.mp3").exists()
    assert final_audio.stat().st_size > 0
    assert preview_path.read_bytes() == b"preview"
    assert mux_calls == [{"video": job.input_video, "audio": final_audio, "subtitles": srt_path, "output": preview_path}]
    sync_report = json.loads(sync_report_path.read_text(encoding="utf-8"))
    assert sync_report["segments"][0]["segment_id"] == "m-0001"
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert status["render"]["details"]["final_audio"] == str(final_audio)
    assert status["render"]["details"]["preview"] == str(preview_path)
```

Replace `test_resume_tts_and_render_keeps_srt_when_tts_segment_fails()` with:

```python
def test_resume_tts_and_render_keeps_srt_when_tts_segment_fails(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})
    _patch_resume_media(monkeypatch)

    async def fail_synthesize_segment(self, row, output):
        raise RuntimeError("tts failed")

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fail_synthesize_segment)

    try:
        resume_tts_and_render(job, _resume_settings())
    except RuntimeError as exc:
        assert "No valid TTS segment audio" in str(exc)
    else:
        raise AssertionError("expected final audio assembly failure")

    srt_path = tmp_path / "job" / "output" / "subtitles_vi.srt"
    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    report = (tmp_path / "job" / "tts" / "tts_warnings.json").read_text(encoding="utf-8")
    assert "m-0001" in report
    sync_report = json.loads((tmp_path / "job" / "tts" / "sync_report.json").read_text(encoding="utf-8"))
    assert "missing TTS audio" in sync_report["segments"][0]["warnings"][0]
```

Replace `test_resume_tts_clears_stale_warning_count_after_success()` with:

```python
def test_resume_tts_clears_stale_warning_count_after_success(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    warning_path = job_root / "tts" / "tts_warnings.json"
    warning_path.parent.mkdir(parents=True)
    warning_path.write_text(json.dumps([{"segment_id": "old", "error": "old failure"}]), encoding="utf-8")
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    _patch_resume_media(monkeypatch)

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fake_synthesize_segment)

    resume_tts_and_render(job, _resume_settings())

    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["tts"]["details"]["warnings"] == 0
    assert status["render"]["details"]["preview"] == str(job_root / "output" / "preview_vi.mp4")
    assert not warning_path.exists()
```

- [ ] **Step 2: Run pipeline tests to verify failure**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_resume_tts_and_render_writes_output_srt tests/test_pipeline.py::test_resume_tts_and_render_keeps_srt_when_tts_segment_fails tests/test_pipeline.py::test_resume_tts_clears_stale_warning_count_after_success -v
```

Expected: FAIL because `resume_tts_and_render()` still writes an empty `tts/final_vi.wav` and does not create `tts/sync_report.json` or `output/preview_vi.mp4`.

- [ ] **Step 3: Integrate assembly and muxing in pipeline**

Add this helper to `src/vietdub/pipeline.py` after `_validate_resume_segment_ids()`:

```python
def _probe_video_duration_ms(job: Job) -> int | None:
    from .media import probe_media

    try:
        metadata = probe_media(job.input_video)
    except RuntimeError:
        return None
    duration = metadata.get("format", {}).get("duration")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        return None
    if duration_seconds <= 0:
        return None
    return int(duration_seconds * 1000)
```

Replace `resume_tts_and_render()` in `src/vietdub/pipeline.py` with:

```python
def resume_tts_and_render(job: Job, settings) -> Path:
    import asyncio

    from .media import mux_preview
    from .srt import render_srt
    from .sync import assemble_final_audio
    from .translate import import_review_csv
    from .tts import EdgeTtsEngine

    rows = import_review_csv(job.root / "translation" / "review.csv")
    _validate_resume_segment_ids(job, rows)
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
    job.mark_done(StepName.RENDER, {"subtitles": str(srt_path), "segments": len(vietnamese_segments)})

    async def synthesize_all() -> int:
        engine = EdgeTtsEngine(settings.edge_voice)
        warnings: list[dict[str, str]] = []
        warning_path = job.root / "tts" / "tts_warnings.json"
        for row in rows:
            if row.status == "skip" or not row.text_vi.strip():
                continue
            try:
                await engine.synthesize_segment(row, job.root / "tts" / "segments" / f"{row.segment_id}.mp3")
            except Exception as exc:  # noqa: BLE001 - keep subtitle output even if one TTS request fails.
                warnings.append({"segment_id": row.segment_id, "error": str(exc)})
        if warnings:
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.write_text(json.dumps(warnings, ensure_ascii=False, indent=2), encoding="utf-8")
        elif warning_path.exists():
            warning_path.unlink()
        return len(warnings)

    warning_count = asyncio.run(synthesize_all())
    job.mark_done(StepName.TTS, {"segments": len(vietnamese_segments), "warnings": warning_count})

    final_audio = job.root / "tts" / "final_vi.wav"
    sync_report_path = job.root / "tts" / "sync_report.json"
    sync_report = assemble_final_audio(
        rows=rows,
        segment_dir=job.root / "tts" / "segments",
        output_audio=final_audio,
        sample_rate=settings.sample_rate,
        video_duration_ms=_probe_video_duration_ms(job),
        sync_report_path=sync_report_path,
    )
    if not final_audio.exists() or final_audio.stat().st_size == 0:
        raise RuntimeError(f"Final audio was not created: {final_audio}")

    preview = job.root / "output" / "preview_vi.mp4"
    mux_preview(job.input_video, final_audio, srt_path, preview)
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
```

- [ ] **Step 4: Run focused pipeline tests to verify pass**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_resume_tts_and_render_writes_output_srt tests/test_pipeline.py::test_resume_tts_and_render_keeps_srt_when_tts_segment_fails tests/test_pipeline.py::test_resume_tts_clears_stale_warning_count_after_success -v
```

Expected: PASS for the three focused resume tests.

- [ ] **Step 5: Run all pipeline tests**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: PASS for every test in `tests/test_pipeline.py`.

- [ ] **Step 6: Commit pipeline integration**

Run:

```powershell
git add tests/test_pipeline.py src/vietdub/pipeline.py
git commit -m "feat: mux dubbed preview on resume"
```

---

### Task 3: Regression And Acceptance Check

**Files:**
- Verify: `src/vietdub/sync.py`
- Verify: `src/vietdub/pipeline.py`
- Verify: `tests/test_sync.py`
- Verify: `tests/test_pipeline.py`

- [ ] **Step 1: Run focused audio and resume tests**

Run:

```powershell
python -m pytest tests/test_sync.py tests/test_pipeline.py -v
```

Expected: PASS for all sync and pipeline tests.

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m pytest
```

Expected: PASS for the full suite.

- [ ] **Step 3: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only user-owned untracked paths remain if they were present before implementation, such as `data/` or `docs/superpowers/plans/2026-06-23-vietdub-cli.md`. No modified implementation or test files should remain unstaged after the task commits.

- [ ] **Step 4: Confirm acceptance artifacts in a local fake job test**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_resume_tts_and_render_writes_output_srt -v
```

Expected: PASS, proving that `resume_tts_and_render()` creates `output/subtitles_vi.srt`, non-empty `tts/final_vi.wav`, `tts/sync_report.json`, and `output/preview_vi.mp4` through the mocked mux path.

---

## Spec Coverage Check

- Final audio assembly: Task 1 implements `assemble_final_audio()` and verifies non-empty WAV output.
- Segment stretching/compression: Task 1 uses `speed_factor_for_duration()` and duration fitting, with tests that force both shorter and longer source clips into target slots.
- Segment warnings: Task 1 verifies missing and zero-byte segment warnings; Task 2 verifies synthesis failure produces TTS warnings and sync report warnings.
- Sync report: Task 1 writes and asserts `tts/sync_report.json` shape; Task 2 asserts the pipeline path.
- Muxed preview: Task 2 verifies `mux_preview()` is called after non-empty final audio and creates `output/preview_vi.mp4`.
- Fatal no-valid-audio behavior: Task 1 verifies assembly failure; Task 2 verifies resume preserves SRT/TTS warnings before surfacing the clear failure.
- Regression: Task 3 runs focused and full pytest commands.
