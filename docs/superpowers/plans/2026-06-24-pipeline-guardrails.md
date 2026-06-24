# Pipeline Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current VietDub CLI pipeline safer and easier to recover by guarding user/LLM/job-memory inputs, persisting artifacts as each step completes, and reporting useful status.

**Architecture:** Keep the existing module boundaries. `pipeline.py` remains the orchestrator; `translate.py` owns LLM and review CSV parsing; `memory.py` owns best-effort historical memory scanning; `cli.py` owns user-facing command errors. No step-runner, database, or final audio muxing work is introduced.

**Tech Stack:** Python 3.11, Typer/Click, Pydantic v2, pytest, JSON/CSV standard library, pathlib.

---

## File Structure

- Modify: `src/vietdub/translate.py`
  - Add a testable LLM response parser.
  - Make `import_review_csv()` fail with clear `RuntimeError` messages for bad headers or rows.
- Modify: `src/vietdub/memory.py`
  - Add safe confidence parsing for old job memory.
  - Record malformed confidence values as `MemoryWarningItem` and skip only the bad entry.
- Modify: `src/vietdub/pipeline.py`
  - Persist STT/OCR/merged/context artifacts immediately.
  - Mark status for main steps.
  - Validate review row segment IDs before writing TTS segment files.
- Modify: `src/vietdub/cli.py`
  - Convert resume/runtime validation failures to `ClickException`.
  - Use configured `JOBS_DIR` for `inspect`.
- Modify: `tests/test_translate.py`
  - Cover LLM response parser and CSV validation failures.
- Modify: `tests/test_memory.py`
  - Cover malformed memory confidence warnings.
- Modify: `tests/test_pipeline.py`
  - Cover artifact persistence before translation failure and unsafe segment IDs.
- Modify: `tests/test_cli.py`
  - Cover configured jobs directory use and clean resume errors.

---

### Task 1: Translate Module Data Errors

**Files:**
- Modify: `tests/test_translate.py`
- Modify: `src/vietdub/translate.py`

- [ ] **Step 1: Add failing tests for LLM response parsing and review CSV errors**

Append these tests to `tests/test_translate.py`:

```python
import pytest

from vietdub.translate import parse_translation_response


def test_parse_translation_response_rejects_invalid_json():
    with pytest.raises(RuntimeError, match="Invalid LLM translation response"):
        parse_translation_response("{bad json")


def test_parse_translation_response_requires_translations_list():
    with pytest.raises(RuntimeError, match="translations"):
        parse_translation_response('{"items": []}')


def test_parse_translation_response_identifies_bad_item():
    with pytest.raises(RuntimeError, match="translation item 1"):
        parse_translation_response('{"translations": ["not an object"]}')


def test_import_review_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("segment_id,start_ms,end_ms,text_cn,text_vi\nm-0001,0,1000,\u4f60\u597d,Xin chao\n", encoding="utf-8-sig")

    with pytest.raises(RuntimeError, match="missing columns"):
        import_review_csv(path)


def test_import_review_csv_reports_invalid_row_number(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,1000,0,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(RuntimeError, match="row 2"):
        import_review_csv(path)
```

- [ ] **Step 2: Run the new translate tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_translate.py -v
```

Expected: FAIL because `parse_translation_response` is not defined and `import_review_csv()` does not yet wrap header/row errors.

- [ ] **Step 3: Implement LLM parser and CSV error wrapping**

Edit `src/vietdub/translate.py`.

Add imports near the top:

```python
from json import JSONDecodeError

from pydantic import ValidationError
```

Add this helper after `ALLOWED_REVIEW_STATUSES`:

```python
def parse_translation_response(content: str) -> list[TranslationRow]:
    try:
        data = json.loads(content)
    except JSONDecodeError as exc:
        raise RuntimeError(f"Invalid LLM translation response: invalid JSON at char {exc.pos}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Invalid LLM translation response: expected a JSON object")

    translations = data.get("translations")
    if not isinstance(translations, list):
        raise RuntimeError("Invalid LLM translation response: expected 'translations' to be a list")

    rows: list[TranslationRow] = []
    for index, item in enumerate(translations, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid LLM translation response: translation item {index} is not an object")
        if item.get("status") not in ALLOWED_REVIEW_STATUSES:
            item = {**item, "status": "draft"}
        try:
            rows.append(TranslationRow.model_validate(item))
        except ValidationError as exc:
            raise RuntimeError(f"Invalid LLM translation response: translation item {index} failed validation") from exc
    return rows
```

Replace the final parsing block in `translate_with_llm()`:

```python
    data = json.loads(content)
    rows: list[TranslationRow] = []
    for item in data["translations"]:
        if item.get("status") not in ALLOWED_REVIEW_STATUSES:
            item["status"] = "draft"
        rows.append(TranslationRow.model_validate(item))
    return rows
```

with:

```python
    return parse_translation_response(content)
```

Replace `import_review_csv()` with:

```python
def import_review_csv(path: Path) -> list[TranslationRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in CSV_FIELDS if field not in fieldnames]
        if missing:
            raise RuntimeError(f"Invalid review CSV {path}: missing columns: {', '.join(missing)}")

        rows: list[TranslationRow] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(TranslationRow.model_validate(row))
            except ValidationError as exc:
                raise RuntimeError(f"Invalid review CSV {path}: row {row_number} failed validation") from exc
        return rows
```

- [ ] **Step 4: Run translate tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_translate.py tests/test_translate_responses.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit translate error handling**

Run:

```powershell
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "fix: report translation data errors clearly"
```

Expected: commit succeeds and only those two files are included.

---

### Task 2: System Memory Malformed Confidence

**Files:**
- Modify: `tests/test_memory.py`
- Modify: `src/vietdub/memory.py`

- [ ] **Step 1: Add failing memory tests for bad confidence values**

Append these tests to `tests/test_memory.py`:

```python
def test_collect_system_memory_skips_character_with_bad_confidence(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "old"
    (job / "context").mkdir(parents=True)
    (job / "context" / "characters.json").write_text(
        json.dumps(
            [
                {"name_cn": "\u574f", "name_vi": "Hong", "confidence": "bad"},
                {"name_cn": "\u597d", "name_vi": "Tot", "confidence": 0.9},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = collect_system_memory(jobs_dir)

    assert [item.name_cn for item in memory.characters] == ["\u597d"]
    assert len(memory.warnings) == 1
    assert "characters.json" in memory.warnings[0].path
    assert "confidence" in memory.warnings[0].message


def test_collect_system_memory_skips_glossary_entry_with_bad_confidence(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "old"
    (job / "context").mkdir(parents=True)
    (job / "context" / "glossary.json").write_text(
        json.dumps(
            {
                "\u574f": {"target": "hong", "confidence": "bad"},
                "\u597d": {"target": "tot", "confidence": 0.9},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = collect_system_memory(jobs_dir)

    assert [item.source_text for item in memory.glossary] == ["\u597d"]
    assert len(memory.warnings) == 1
    assert "glossary.json" in memory.warnings[0].path
    assert "confidence" in memory.warnings[0].message
```

- [ ] **Step 2: Run memory tests and verify the new tests fail**

Run:

```powershell
python -m pytest tests/test_memory.py -v
```

Expected: FAIL with `ValueError` from `float(...)`.

- [ ] **Step 3: Implement safe confidence parsing**

Edit `src/vietdub/memory.py`.

Add this import:

```python
import math
```

Add this helper before `_glossary_target_and_confidence()`:

```python
def _parse_confidence(
    raw_value: Any,
    default_confidence: float,
    path: Path,
    warnings: list[MemoryWarningItem],
    label: str,
) -> float | None:
    if raw_value is None or raw_value == "":
        return default_confidence
    try:
        confidence = float(raw_value)
    except (TypeError, ValueError) as exc:
        warnings.append(MemoryWarningItem(path=str(path), message=f"Invalid confidence for {label}: {raw_value!r}"))
        return None
    if not math.isfinite(confidence):
        warnings.append(MemoryWarningItem(path=str(path), message=f"Invalid confidence for {label}: {raw_value!r}"))
        return None
    return confidence
```

In `_collect_characters_json()`, replace:

```python
        _put_character(
            characters,
            MemoryCharacter(
                name_cn=name_cn,
                name_vi=name_vi,
                source_job=job_dir.name,
                source="context/characters.json",
                confidence=float(item.get("confidence") or 0.80),
            ),
        )
```

with:

```python
        confidence = _parse_confidence(item.get("confidence"), 0.80, path, warnings, name_cn)
        if confidence is None:
            continue
        _put_character(
            characters,
            MemoryCharacter(
                name_cn=name_cn,
                name_vi=name_vi,
                source_job=job_dir.name,
                source="context/characters.json",
                confidence=confidence,
            ),
        )
```

Replace `_glossary_target_and_confidence()` with:

```python
def _glossary_target_and_confidence(
    raw_target: Any,
    default_confidence: float,
    path: Path,
    warnings: list[MemoryWarningItem],
    label: str,
) -> tuple[str, float | None]:
    if isinstance(raw_target, dict):
        target = str(raw_target.get("target") or raw_target.get("name_vi") or "").strip()
        confidence = _parse_confidence(raw_target.get("confidence"), default_confidence, path, warnings, label)
        return target, confidence
    return str(raw_target or "").strip(), default_confidence
```

In `_collect_glossary_json()`, replace:

```python
        target, confidence = _glossary_target_and_confidence(raw_target, 0.80)
        if not source or not target:
            continue
```

with:

```python
        target, confidence = _glossary_target_and_confidence(raw_target, 0.80, path, warnings, source)
        if not source or not target or confidence is None:
            continue
```

- [ ] **Step 4: Run memory tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_memory.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit memory guardrail**

Run:

```powershell
git add src/vietdub/memory.py tests/test_memory.py
git commit -m "fix: skip malformed system memory confidence"
```

Expected: commit succeeds and only memory files are included.

---

### Task 3: Persist Pipeline Artifacts Before Translation

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/vietdub/pipeline.py`

- [ ] **Step 1: Add failing test for translation failure artifact persistence**

Append this test to `tests/test_pipeline.py`:

```python
def test_review_pipeline_persists_artifacts_before_translate_failure(monkeypatch, tmp_path):
    from vietdub.pipeline import run_review_pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")

    def fake_extract_audio(video_path, audio_path, sample_rate):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")

    class FakeSttEngine:
        def __init__(self, model, language):
            self.model = model
            self.language = language

        def transcribe(self, audio_path):
            return [TimedSegment(id="s-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d", source="stt")]

    class FakeOcrEngine:
        def recognize(self, video_path):
            return [TimedSegment(id="o-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d", source="ocr")]

    def fail_translate(segments, context_bundle, api_key, model, base_url):
        raise RuntimeError("translation failed")

    monkeypatch.setattr("vietdub.media.extract_audio", fake_extract_audio)
    monkeypatch.setattr("vietdub.stt.FasterWhisperSttEngine", FakeSttEngine)
    monkeypatch.setattr("vietdub.ocr.PaddleSubtitleOcrEngine", FakeOcrEngine)
    monkeypatch.setattr("vietdub.reference.build_reference_context", lambda segments, data_dir: {})
    monkeypatch.setattr("vietdub.translate.translate_with_llm", fail_translate)

    settings = type(
        "Settings",
        (),
        {
            "sample_rate": 44100,
            "stt_model": "tiny",
            "stt_language": "zh",
            "reference_data_dir": str(tmp_path / "data"),
            "openai_api_key": "key",
            "llm_model": "model",
            "openai_base_url": "",
        },
    )()

    try:
        run_review_pipeline(video=video, jobs_dir=tmp_path / "jobs", series=None, settings=settings)
    except RuntimeError as exc:
        assert "translation failed" in str(exc)
    else:
        raise AssertionError("expected translation failure")

    job_root = tmp_path / "jobs" / "clip"
    assert (job_root / "stt" / "segments.json").exists()
    assert (job_root / "ocr" / "subtitles.json").exists()
    assert (job_root / "transcript" / "merged.json").exists()
    assert (job_root / "context" / "style_guide.json").exists()
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["extract"]["state"] == "done"
    assert status["stt"]["state"] == "done"
    assert status["ocr"]["state"] == "done"
    assert status["merge"]["state"] == "done"
    assert status["context"]["state"] == "done"
    assert "translate" not in status
```

- [ ] **Step 2: Run the new pipeline test and verify it fails**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_review_pipeline_persists_artifacts_before_translate_failure -v
```

Expected: FAIL because artifacts are written after `translate_with_llm()`.

- [ ] **Step 3: Reorder artifact writes and status updates**

Edit `src/vietdub/pipeline.py`.

In `run_review_pipeline()`, replace the block from STT through translate writes with this sequence:

```python
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
        typed_settings.openai_api_key,
        typed_settings.llm_model,
        typed_settings.openai_base_url,
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
```

Remove the old duplicate writes that happen after the translate call.

- [ ] **Step 4: Run pipeline tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit pipeline persistence**

Run:

```powershell
git add src/vietdub/pipeline.py tests/test_pipeline.py
git commit -m "fix: persist pipeline artifacts before translation"
```

Expected: commit succeeds and only pipeline files are included.

---

### Task 4: Safe TTS Segment IDs And Resume Status

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/vietdub/pipeline.py`

- [ ] **Step 1: Add failing tests for unsafe segment IDs**

Append these tests to `tests/test_pipeline.py`:

```python
def test_resume_tts_rejects_path_traversal_segment_id(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "..\\evil,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})

    async def fake_synthesize_segment(self, row, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp3")
        return output

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fake_synthesize_segment)

    try:
        resume_tts_and_render(job, type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural"})())
    except RuntimeError as exc:
        assert "Invalid segment_id" in str(exc)
    else:
        raise AssertionError("expected invalid segment_id failure")

    assert not (tmp_path / "job" / "evil.mp3").exists()
    assert not (tmp_path / "evil.mp3").exists()


def test_resume_tts_rejects_unknown_transcript_segment_id(tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-9999,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": "m-0001", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})

    try:
        resume_tts_and_render(job, type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural"})())
    except RuntimeError as exc:
        assert "m-9999" in str(exc)
    else:
        raise AssertionError("expected unknown segment_id failure")
```

- [ ] **Step 2: Run the new TTS safety tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_resume_tts_rejects_path_traversal_segment_id tests/test_pipeline.py::test_resume_tts_rejects_unknown_transcript_segment_id -v
```

Expected: FAIL because resume currently uses `segment_id` directly in the output path.

- [ ] **Step 3: Add segment ID helpers**

Edit `src/vietdub/pipeline.py`.

Add imports at the top:

```python
import re
```

Add this module constant after imports:

```python
SAFE_SEGMENT_ID_RE = re.compile(r"^m-\d+$")
```

Add these helpers before `resume_tts_and_render()`:

```python
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


def _is_safe_fallback_segment_id(segment_id: str) -> bool:
    if not segment_id:
        return False
    if "/" in segment_id or "\\" in segment_id or ":" in segment_id:
        return False
    return SAFE_SEGMENT_ID_RE.fullmatch(segment_id) is not None


def _validate_resume_segment_ids(job: Job, rows: list[TranslationRow]) -> None:
    allowed_ids = _load_allowed_segment_ids(job)
    review_path = job.root / "translation" / "review.csv"
    for row in rows:
        if row.status == "skip" or not row.text_vi.strip():
            continue
        segment_id = row.segment_id
        if allowed_ids is not None:
            if segment_id not in allowed_ids:
                raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r} is not in transcript/merged.json")
            continue
        if not _is_safe_fallback_segment_id(segment_id):
            raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
```

- [ ] **Step 4: Use the helpers in resume and mark status**

In `resume_tts_and_render()`, after:

```python
    rows = import_review_csv(job.root / "translation" / "review.csv")
```

add:

```python
    _validate_resume_segment_ids(job, rows)
```

After writing the SRT:

```python
    srt_path.write_text(render_srt(vietnamese_segments), encoding="utf-8")
```

add:

```python
    job.mark_done(StepName.RENDER, {"subtitles": str(srt_path), "segments": len(vietnamese_segments)})
```

After:

```python
    asyncio.run(synthesize_all())
```

add this block:

```python
    warnings_path = job.root / "tts" / "tts_warnings.json"
    warning_count = 0
    if warnings_path.exists():
        try:
            warning_count = len(json.loads(warnings_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            warning_count = 1
    job.mark_done(StepName.TTS, {"segments": len(vietnamese_segments), "warnings": warning_count})
```

- [ ] **Step 5: Run pipeline tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit TTS segment guardrail**

Run:

```powershell
git add src/vietdub/pipeline.py tests/test_pipeline.py
git commit -m "fix: reject unsafe tts segment ids"
```

Expected: commit succeeds and only pipeline files are included.

---

### Task 5: CLI Error Boundaries And Configured Inspect Directory

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/vietdub/cli.py`

- [ ] **Step 1: Add failing CLI tests**

Append these tests to `tests/test_cli.py`:

```python
def test_resume_runtime_error_is_click_error(monkeypatch, tmp_path):
    from vietdub import cli
    from vietdub.jobs import Job

    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = Job(root=job_root, config={})

    monkeypatch.setattr(cli, "_open_job", lambda job_dir, jobs_dir: job)

    def fail_resume(job_arg, settings):
        raise RuntimeError("bad review csv")

    monkeypatch.setattr(cli, "resume_tts_and_render", fail_resume)

    result = runner.invoke(app, ["resume", str(job_root), "--from", "tts"])

    assert result.exit_code != 0
    assert "bad review csv" in result.output
    assert "Traceback" not in result.output


def test_inspect_uses_configured_jobs_dir(monkeypatch, tmp_path):
    from vietdub import cli
    from vietdub.jobs import Job

    captured = {}
    configured_jobs_dir = tmp_path / "configured-jobs"

    def fake_open_job(job_dir, jobs_dir):
        captured["job_dir"] = job_dir
        captured["jobs_dir"] = jobs_dir
        return Job(root=job_dir, config={})

    monkeypatch.setenv("JOBS_DIR", str(configured_jobs_dir))
    monkeypatch.setattr(cli, "_open_job", fake_open_job)

    result = runner.invoke(app, ["inspect", "clip"])

    assert result.exit_code == 0
    assert captured["jobs_dir"] == configured_jobs_dir
```

- [ ] **Step 2: Run CLI tests and verify the new tests fail**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: FAIL because `resume` does not catch runtime errors and `inspect` still passes `Path("jobs")`.

- [ ] **Step 3: Wrap resume runtime errors and fix inspect settings**

Edit `src/vietdub/cli.py`.

Replace the `resume()` body with:

```python
def resume(
    job_dir: Path,
    from_step: StepName = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    settings = Settings()
    job = _open_job(job_dir, Path(settings.jobs_dir))
    if from_step == StepName.TTS:
        try:
            srt_path = resume_tts_and_render(job, settings)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None
        safe_echo(f"Vietnamese subtitles written: {srt_path}")
        safe_echo("TTS segment files written under tts/segments.")
        return
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    safe_echo(f"Resume order: {steps}")
```

Replace `inspect()` with:

```python
@app.command()
def inspect(job_dir: Path) -> None:
    settings = Settings()
    job = _open_job(job_dir, Path(settings.jobs_dir))
    safe_echo(job.root)
    safe_echo(job.load_status())
```

- [ ] **Step 4: Run CLI tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit CLI guardrails**

Run:

```powershell
git add src/vietdub/cli.py tests/test_cli.py
git commit -m "fix: report cli runtime errors cleanly"
```

Expected: commit succeeds and only CLI files are included.

---

### Task 6: Full Regression And Acceptance Check

**Files:**
- Verify all modified files.
- No planned source modifications in this task unless a previous test exposes a concrete defect.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
python -m pytest
```

Expected: PASS with all tests passing.

- [ ] **Step 2: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only pre-existing untracked paths may remain:

```text
?? data/
?? docs/superpowers/plans/2026-06-23-vietdub-cli.md
```

If the current plan file is uncommitted, include it in the final documentation commit in Step 4.

- [ ] **Step 3: Review acceptance criteria manually**

Open `docs/superpowers/specs/2026-06-24-pipeline-guardrails-design.md` and confirm each acceptance criterion maps to passing tests:

```text
pytest passes -> Step 1
unsafe segment_id blocked -> Task 4 tests
translate failure leaves artifacts -> Task 3 test
bad memory confidence warns -> Task 2 tests
LLM/CSV errors readable -> Task 1 tests and Task 5 CLI boundary
inspect uses configured jobs dir -> Task 5 test
```

- [ ] **Step 4: Commit this implementation plan if it is still uncommitted**

Run:

```powershell
git add docs/superpowers/plans/2026-06-24-pipeline-guardrails.md
git commit -m "docs: plan pipeline guardrails"
```

Expected: commit succeeds if the plan file was not committed before execution. If Git reports nothing to commit for this file, continue.

- [ ] **Step 5: Final status report**

Run:

```powershell
git log --oneline -6
git status --short
```

Expected: recent commits include the guardrail tasks, and status contains no unexpected modified tracked files.

Final response should include:

```text
Implemented pipeline guardrails.
Tests: python -m pytest
Commits: <list commit hashes and subjects>
Remaining untracked files: data/, docs/superpowers/plans/2026-06-23-vietdub-cli.md
```
