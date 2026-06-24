# VietDub Final Audio Preview Design

## Summary

This design adds the missing final audio assembly step to VietDub. After `resume --from tts`, the pipeline should produce a timed Vietnamese audio track at `tts/final_vi.wav` and mux it with the input video and rendered subtitles into `output/preview_vi.mp4`.

Chosen approach: use `pydub` to build a silent timeline in Python, place each valid TTS segment at its subtitle start time, and use existing FFmpeg helpers for stretch/mux work where appropriate.

## Scope

In scope:

- Assemble `tts/segments/*.mp3` into `tts/final_vi.wav`.
- Stretch/compress each segment to fit its original subtitle duration.
- Skip bad segment audio with warnings instead of failing the whole resume.
- Write a sync report for per-segment timing and warning details.
- Mux `input.mp4`, `tts/final_vi.wav`, and `output/subtitles_vi.srt` into `output/preview_vi.mp4`.
- Add focused tests for audio assembly, warnings, and resume integration.

Out of scope:

- Improving OCR duplicate detection.
- Changing translation prompts or review CSV format.
- Replacing Edge TTS.
- Full multi-step resume beyond the existing `--from tts` path.
- Advanced loudness normalization or voice mixing.

## Current Problem

`resume_tts_and_render()` currently writes subtitle output and individual MP3 files, then creates an empty `tts/final_vi.wav`. Because the final audio file is empty, `mux_preview()` is never called and `preview_vi.mp4` is not created.

The result is useful for review and partial TTS verification, but not enough for a dubbed preview.

## Architecture

Keep existing ownership boundaries:

- `tts.py`: synthesize individual MP3 files with Edge TTS.
- `sync.py`: provide audio timing utilities, stretch helpers, and final timeline assembly.
- `pipeline.py`: orchestrate resume, write warnings/status, call final audio assembly, and call mux.
- `media.py`: continue owning FFmpeg media extraction/probing/muxing.
- `models.py`: reuse `SyncReport` and `SyncSegmentReport` for report output.

No new service, database, or UI is introduced.

## Resume Data Flow

The revised `resume_tts_and_render()` flow:

```text
read translation/review.csv
  -> validate segment ids
  -> render output/subtitles_vi.srt
  -> synthesize tts/segments/<segment_id>.mp3
  -> assemble final audio:
       create silent timeline
       load each valid MP3
       stretch/compress to row duration
       overlay at row.start_ms
       collect per-segment sync report
  -> write tts/final_vi.wav
  -> write tts/sync_report.json
  -> mux input.mp4 + final_vi.wav + subtitles_vi.srt
  -> write output/preview_vi.mp4
```

If no valid segment audio is available, final audio assembly should fail with a clear `RuntimeError`.

## Audio Assembly Behavior

Input rows:

- Use rows with non-empty `text_vi` and `status != "skip"`.
- Use `start_ms` and `end_ms` from `review.csv`.
- Target duration is `end_ms - start_ms`; invalid durations are skipped with warnings.

Audio loading:

- Load `tts/segments/<segment_id>.mp3` for each active row.
- Missing, zero-byte, or unreadable files are skipped with warnings.
- Successful files are converted into a format suitable for WAV export.

Timing:

- Each loaded segment is stretched or compressed to fit its target duration.
- Speed factor is recorded as `actual_duration_ms / target_duration_ms`.
- Segment audio is overlaid onto a silent timeline at `start_ms`.

Timeline length:

- Prefer the video duration from `probe_media(job.input_video)`.
- If media probing fails or duration is absent, use the maximum `end_ms` among active rows.
- The final timeline should be at least as long as the last overlaid subtitle segment.

Output:

- Write `tts/final_vi.wav`.
- Write `tts/sync_report.json`.
- Call `mux_preview()` only after `final_vi.wav` exists and has non-zero size.

## Stretch Strategy

Reuse the current sync concepts:

- `speed_factor_for_duration(actual_ms, target_ms)` computes the speed factor.
- `atempo_filters(factor)` remains the source of FFmpeg-compatible atempo chains.

Implementation may use either:

- FFmpeg `atempo` on temporary files, then load the stretched WAV with `pydub`; or
- pydub speed manipulation if it produces stable durations.

The implementation should favor testability and Windows path reliability over minimizing temporary files.

## Error Handling

Segment-level failures:

- Do not fail the whole resume for one bad segment.
- Add a warning to the sync report and continue.
- Existing TTS warnings from synthesis should remain available.

Fatal failures:

- If no valid audio segments can be assembled, raise `RuntimeError`.
- If `mux_preview()` fails, let the `RuntimeError` surface to CLI handling.
- If timeline export fails, raise `RuntimeError`.

Status:

- Keep `render` status after SRT write.
- Keep `tts` status after synthesis.
- Add sync/mux details to status when final audio and preview are created. Store them under existing `render` or `tts` status details unless the implementation plan explicitly justifies a new `StepName`.

## Reports

Write `tts/sync_report.json` using the existing `SyncReport` model:

- `output_audio`: path to `tts/final_vi.wav`.
- `sample_rate`: configured sample rate.
- `segments`: one item per active row, including skipped/error cases where practical.

Each segment report should include:

- `segment_id`
- `start_ms`
- `end_ms`
- `original_duration_ms`
- `synced_duration_ms`
- `speed_factor`
- `warnings`

For missing/unreadable segment files, use `0` for unavailable durations and add a warning.

## Testing

Add focused tests:

- Audio assembly creates `final_vi.wav` from two short segment files and preserves total timeline duration.
- Stretched segment duration is close to target duration.
- Missing or zero-byte segment file is recorded as warning and skipped.
- Assembly fails clearly when no valid segment audio exists.
- Resume integration with fake TTS creates `subtitles_vi.srt`, `tts/final_vi.wav`, `tts/sync_report.json`, and calls `mux_preview()`.
- Existing pipeline/CLI tests continue to pass.

Tests should avoid network calls. Use generated local audio fixtures and monkeypatch Edge TTS/muxing where needed.

## Risks

- `pydub` loads the full timeline into memory. This is acceptable for MVP previews but may need FFmpeg-only assembly for long videos later.
- Speed adjustment can change voice quality. The MVP prioritizes timing over perfect naturalness.
- OCR duplicates can produce repeated TTS lines. This design intentionally does not solve duplicate detection; it assembles whatever `review.csv` contains.

## Acceptance Criteria

- `python -m pytest` passes.
- `vietdub resume <job> --from tts` creates a non-empty `tts/final_vi.wav` when at least one valid TTS segment exists.
- `vietdub resume <job> --from tts` creates `output/preview_vi.mp4` when muxing succeeds.
- `tts/sync_report.json` is written and records per-segment timing/warnings.
- Bad individual MP3 files do not prevent valid segments from appearing in final audio.
- A job with no valid segment audio fails with a clear CLI error.
