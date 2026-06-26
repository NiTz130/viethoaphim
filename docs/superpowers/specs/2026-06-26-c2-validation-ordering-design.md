# C2: Reorder validation checks in `_validate_resume_segment_ids`

**Date:** 2026-06-26
**Scope:** Single function + one new test
**Status:** Approved (design), pending implementation

## Context

`src/vietdub/pipeline.py:_validate_resume_segment_ids` (lines 152-174) validates segment IDs from `review.csv` before resuming TTS + render. It runs checks in this order:

1. Timing
2. Duplicate
3. Add to `active_ids`
4. Safety (`_is_safe_segment_path_component`)
5. Membership (if transcript exists)
6. Fallback safety (if no transcript)

When a transcript-injected unsafe ID (e.g. `..\evil`) appears **more than once** in `review.csv`, the first occurrence fails the safety check (correct), but the second occurrence is caught by the **duplicate** check first and reports `"Duplicate segment_id"` — masking the real problem. The user fixes the apparent duplicate but the underlying unsafe ID is still present.

This was flagged as **C2** in the project review (2026-06-26).

## Goal

Make the safety check the first per-row validation step so unsafe IDs always surface as `"Invalid segment_id"`, regardless of whether they also duplicate.

## Non-goals

- H4 (translation status downgrade warning) — separate spec
- M8 (translate error-path tests) — separate spec
- Collecting multiple errors before raising — rejected at approach question
- Changing the set of error messages — same wording, only their order changes

## Design

### Code change

Reorder the per-row checks in `_validate_resume_segment_ids`:

**New order:**

1. Timing check (unchanged position — different concern)
2. Safety check (`_is_safe_segment_path_component`) — **moved up**
3. Membership check (if `allowed_ids is not None`)
4. Fallback safety check (`_is_safe_fallback_segment_id`) — converted from `if` to `elif`
5. Duplicate check — **moved down**
6. `active_ids.add(segment_id)` — last, after duplicate check

**New control flow:**

```python
for row in rows:
    if row.status == "skip" or not row.text_vi.strip():
        continue
    segment_id = row.segment_id
    if row.start_ms < 0 or row.end_ms <= row.start_ms:
        raise RuntimeError(...)
    if not _is_safe_segment_path_component(segment_id):
        raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
    if allowed_ids is not None:
        if segment_id not in allowed_ids:
            raise RuntimeError(
                f"Invalid segment_id in {review_path}: {segment_id!r} is not in transcript/merged.json"
            )
    elif not _is_safe_fallback_segment_id(segment_id):
        raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
    if segment_id in active_ids:
        raise RuntimeError(f"Duplicate segment_id in {review_path}: {segment_id!r}")
    active_ids.add(segment_id)
```

### Behavior change matrix

| Case | Old message | New message |
|---|---|---|
| Unsafe unique ID | Invalid segment_id | Invalid segment_id (unchanged) |
| Unsafe duplicate (1st) | Invalid segment_id | Invalid segment_id (unchanged) |
| Unsafe duplicate (2nd+) | **Duplicate segment_id** | **Invalid segment_id** (fixed) |
| Safe unknown ID in transcript | is not in transcript | is not in transcript (unchanged) |
| Safe valid duplicate | Duplicate segment_id | Duplicate segment_id (unchanged) |
| status=skip duplicate | (skipped before loop body) | (unchanged) |

## Test plan

### New test

`tests/test_pipeline.py::test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message`

Setup:
- `review.csv` with two rows sharing the same unsafe id `..\evil`
- `transcript/merged.json` containing the same unsafe id (transcript-injected case)
- Valid `input.mp4`

Assert:
- `resume_tts_and_render` raises `RuntimeError`
- Error message contains `"Invalid segment_id"`
- Error message contains `"..\\evil"`
- Error message does **not** contain `"Duplicate"` (locks in the C2 fix)

### Existing tests

All existing tests for `_validate_resume_segment_ids` use unique IDs and are unaffected:

- `test_resume_tts_rejects_path_traversal_segment_id`
- `test_resume_tts_preserves_render_state_when_segment_validation_fails`
- `test_resume_tts_rejects_unknown_transcript_segment_id`
- `test_resume_tts_rejects_transcript_path_traversal_segment_id`
- `test_resume_tts_rejects_transcript_windows_absolute_segment_id`
- `test_resume_tts_and_render_rejects_duplicate_active_segment_id`
- `test_resume_tts_and_render_allows_duplicate_inactive_segment_id`

## Risk

**Low.** Behavior changes only in the unsafe-and-duplicate case, and the new message is more accurate. No security implications — the safety check still runs; it just runs earlier. The fallback safety branch becomes `elif` to avoid re-running it after the membership check succeeds (membership implies the id is in `transcript/merged.json`, where it must already be well-formed).

## Rollback

Single function change, single commit. Revert the commit if CI fails.

## Implementation notes

- No new imports needed.
- No new module dependencies.
- The docstring on `resume_tts_and_render` (added in the C1 fix) is unaffected.
- `default=None` semantics of `_load_allowed_segment_ids` are preserved.
