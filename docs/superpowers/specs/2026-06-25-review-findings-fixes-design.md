# Review Findings Fixes Design

## Goal

Fix the four review findings with narrow, tested changes:

- Prevent LLM translation rows from changing canonical segment timing or source text in `review.csv`.
- Make unsupported `vietdub resume --from` values fail clearly instead of exiting successfully.
- Prevent OCR retries from reusing stale frame images.
- Escape subtitle paths safely enough for FFmpeg filter arguments when paths contain apostrophes.

## Scope

In scope:

- Focused changes in `translate.py`, `cli.py`, `ocr.py`, and `media.py`.
- Regression tests for each reviewed issue.
- No broad pipeline refactor.

Out of scope:

- Full resume support for steps other than `tts`.
- Reworking OCR frame extraction strategy beyond clearing the existing job-local frame directory.
- Replacing FFmpeg command construction with a new media abstraction.
- Changing the `review.csv` column format.

## Current Behavior

`export_review_csv()` currently builds a row lookup from LLM translation output and writes the selected row directly. If the LLM returns a valid `segment_id` with incorrect `start_ms`, `end_ms`, `speaker`, or `text_cn`, those incorrect values are exported to `review.csv`.

`resume` accepts every `StepName`. Only `--from tts` performs actual work; other step values print a resume order and exit successfully.

`PaddleSubtitleOcrEngine.recognize()` writes frames to `video_path.parent / "ocr_frames"` and OCRs all `frame_*.jpg` in that directory. A retry can include stale images from a previous partial or longer extraction.

`build_mux_preview_command()` interpolates the subtitle path into an FFmpeg `subtitles='...'` filter string. The current escaping handles backslashes and drive colons, but not apostrophes in the path.

## Proposed Design

### Canonical Review CSV Export

`TimedSegment` remains the source of truth for immutable review columns:

- `segment_id`
- `start_ms`
- `end_ms`
- `speaker`
- `text_cn`

For matching translation rows, `export_review_csv()` copies only editable translation fields:

- `text_vi`
- `context_note`
- `status`

If there is no matching translation row, export behavior stays the same: the row is created from the segment with empty `text_vi` and default `draft` status.

This keeps the CSV editable where intended while preventing model output from corrupting timing or source transcript data.

### Unsupported Resume Steps

`resume` keeps accepting `StepName` values for CLI parsing, but only `StepName.TTS` is supported. For every other value, the command raises a user-facing Click/Typer error such as:

`Only resume from tts is currently supported.`

This preserves the existing supported workflow and removes the false-success path.

### OCR Frame Cleanup

Before FFmpeg extracts OCR frames, `PaddleSubtitleOcrEngine.recognize()` removes stale `frame_*.jpg` files inside the existing `ocr_frames` directory for that video/job.

The cleanup is intentionally scoped:

- It only targets files matching `frame_*.jpg`.
- It only runs inside `video_path.parent / "ocr_frames"`.
- It leaves unrelated files alone.

After cleanup, the existing FFmpeg command and OCR pass remain unchanged.

### FFmpeg Subtitle Path Escaping

Add a small helper for subtitle filter path escaping. It will normalize Windows separators and escape characters required by the current single-quoted filter expression:

- Backslash path separators become `/`.
- Drive colons are escaped as `\:`.
- Apostrophes are escaped so they do not terminate the quoted filter value.

`build_mux_preview_command()` will use the helper and keep the current command shape.

## Tests

Add focused regression tests:

- `tests/test_translate.py`: an LLM row with wrong timing/source text must still export canonical values from `TimedSegment`, while preserving `text_vi`, `context_note`, and `status`.
- `tests/test_cli.py`: `vietdub resume <job> --from stt` exits non-zero and reports that only `tts` resume is supported.
- `tests/test_engines.py` or a new OCR-focused test: monkeypatch FFmpeg and PaddleOCR, create stale `ocr_frames/frame_*.jpg`, run OCR, and assert stale frames were removed before OCR.
- `tests/test_media_srt.py`: a subtitle path containing an apostrophe produces a filter argument that keeps the apostrophe escaped inside the FFmpeg subtitles expression.

Run the full `pytest -q` suite after implementation.

## Risks

The CSV change may discard non-editable metadata that an LLM previously filled differently from the transcript. That is intentional because timing and source transcript data should not be model-controlled.

The OCR cleanup removes old `frame_*.jpg` files from the job-local OCR frame directory. This is appropriate because those files are generated artifacts for the current OCR run.

The FFmpeg escaping remains tied to the current `subtitles='...'` filter string. If later paths expose more FFmpeg escaping edge cases, a larger command construction change may be needed.
