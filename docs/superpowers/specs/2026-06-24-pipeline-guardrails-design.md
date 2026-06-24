# VietDub Pipeline Guardrails Design

## Summary

This design covers a focused reliability pass for the existing VietDub CLI pipeline. The goal is to fix the reviewed issues without turning the codebase into a new pipeline framework.

Chosen approach: add small guardrails around the current architecture. `pipeline.py` remains the orchestrator, while parsing and validation stay close to the modules that own each data source.

## Scope

In scope:

- Prevent unsafe `segment_id` values from writing TTS files outside `tts/segments`.
- Persist expensive artifacts as soon as each pipeline step completes.
- Make system memory scanning best-effort when old job files contain malformed values.
- Convert LLM and review CSV parsing failures into clear CLI errors.
- Make `status.json` useful for `inspect` by marking the main steps as complete.
- Add focused tests for the guardrails.

Out of scope:

- A full step-runner or fully resumable multi-step pipeline.
- Final audio timeline assembly.
- Guaranteed `preview_vi.mp4` generation while `final_vi.wav` remains empty.
- Major refactors unrelated to the reviewed issues.

## Current Problem

The current implementation works for the happy path, but several boundary conditions can waste long-running work or write unsafe outputs:

- `review.csv` is user-editable and currently controls TTS output filenames directly.
- `run_review_pipeline()` waits until after the LLM call to write STT/OCR/merged/context artifacts.
- `collect_system_memory()` records warnings for some malformed files, but malformed `confidence` values can still crash the scan.
- LLM response and CSV validation errors can surface as raw Python exceptions instead of user-facing CLI errors.
- `status.json` currently records `extract` but not the other major steps.

## Architecture

Keep the existing module boundaries:

- `pipeline.py`: orchestrates run and resume flows, writes artifacts, updates status, validates resume inputs before TTS.
- `translate.py`: owns review CSV import/export and LLM response validation.
- `memory.py`: owns scanning old jobs and converting malformed memory entries into warnings.
- `jobs.py`: owns job status writes.
- `cli.py`: owns conversion from runtime errors to clean CLI errors.

No new framework or persistent database is introduced.

## Run Data Flow

The revised `run_review_pipeline()` writes data immediately after each step:

```text
run video
  -> create job
  -> extract audio
  -> mark extract done
  -> STT
  -> write stt/segments.json
  -> mark stt done
  -> OCR
  -> write ocr/subtitles.json
  -> mark ocr done
  -> merge
  -> write transcript/merged.json
  -> mark merge done
  -> build reference + system memory + context
  -> write context/*.json and warnings
  -> mark context done
  -> translate
  -> write translation/translated.json and translation/review.csv
  -> mark translate done
```

This preserves expensive intermediate output even when translation fails.

## Resume Data Flow

The revised `resume_tts_and_render()` validates review rows before writing segment audio:

```text
resume --from tts
  -> import review.csv with validation
  -> validate segment ids
  -> render output/subtitles_vi.srt
  -> synthesize safe tts/segments/<segment_id>.mp3
  -> write per-segment TTS warnings when needed
  -> mark tts/render status as appropriate
```

Validation should prefer IDs from `transcript/merged.json` when that file exists. If it does not exist, use a strict fallback format such as `m-0001` to preserve fixture/test usability.

## Segment ID Safety

`segment_id` from `review.csv` must not be used as a path component until validated.

Rules:

- Accept IDs matching the known merged transcript IDs when `transcript/merged.json` is present.
- If no transcript is available, accept only `m-` followed by digits, for example `m-0001`.
- Reject values containing path separators, drive prefixes, empty strings, or unknown IDs.
- Fail before writing any TTS files if an invalid ID is found.

The failure message should identify the bad `segment_id` and the review CSV path.

## LLM And CSV Error Handling

`translate_with_llm()` should wrap response parsing and row validation:

- Invalid JSON becomes `RuntimeError("Invalid LLM translation response: ...")`.
- Missing or non-list `translations` becomes a `RuntimeError`.
- Non-object translation items become a `RuntimeError`.
- Pydantic row validation errors become a `RuntimeError` with enough context to identify the item.

`import_review_csv()` should wrap row validation similarly:

- Invalid field values should report the CSV path and row number.
- Missing required columns should fail with a clear message.
- Status handling should remain strict for current job data.

## System Memory Error Handling

Old job data is advisory. It must not prevent a new job from running.

Malformed memory entries should be handled as follows:

- Invalid JSON or unreadable files continue to produce warnings.
- Invalid `confidence` values in `characters.json` or `glossary.json` produce a warning and skip only that entry.
- Unsupported shapes continue to produce warnings when the whole file has the wrong top-level type.
- Valid entries from the same job should still be collected.

## Status Handling

Use the existing `Job.mark_done()` method. Add calls for:

- `extract`
- `stt`
- `ocr`
- `merge`
- `context`
- `translate`
- `tts`
- `render`

Details should be short and useful: output path, row count, segment count, or warning count.

This does not implement step-level resume for every step. It makes `inspect` accurate enough for debugging.

## CLI Behavior

`run` and `resume` should convert pipeline/runtime validation failures into clean Click/Typer errors.

`inspect` should use `Settings().jobs_dir` instead of hard-coded `jobs`, matching `run` and `resume`.

## Testing

Add focused tests:

- TTS resume rejects `segment_id` path traversal and absolute paths before writing files.
- TTS resume accepts valid transcript IDs.
- System memory skips bad confidence entries and records warnings.
- LLM invalid JSON or missing `translations` returns a clear `RuntimeError`.
- Review CSV invalid rows return a clear `RuntimeError` with path and row number.
- `run_review_pipeline()` persists STT/OCR/merged/context artifacts before a translate failure.
- `inspect` uses custom `JOBS_DIR`.

Existing tests should continue to pass.

## Risks

- Stricter review CSV validation may fail jobs that previously limped through malformed rows. This is intentional for current job input because unsafe filenames and invalid timings are worse than a clear early failure.
- Persisting artifacts earlier changes the state left behind after a failed run. This is intentional and should improve recovery/debugging.
- Status updates add more writes to `status.json`, but the files are small and job-local.

## Acceptance Criteria

- `python -m pytest` passes.
- Invalid `segment_id` values cannot write outside the current job directory.
- A translation failure after STT/OCR leaves `stt/segments.json`, `ocr/subtitles.json`, `transcript/merged.json`, and context files available.
- Malformed old memory entries are reported in `context/system_memory_warnings.json` when selected during a run, without crashing the job.
- CLI errors for LLM/CSV validation are readable and do not show raw tracebacks.
- `vietdub inspect` reports status from the configured jobs directory.
