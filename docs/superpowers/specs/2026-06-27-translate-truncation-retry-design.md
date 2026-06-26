# Translate robustness: truncation detection + per-batch retry

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` + new tests in `tests/test_translate.py` (or `tests/test_translate_responses.py`)
**Status:** Approved (design), pending implementation

## Context

The recent batching refactor in `src/vietdub/translate.py` (commits a5d8b33, 09bc18a, 51e350a) split `translate_with_llm` into batched calls (groups of `LLM_BATCH_SIZE = 50`) and raised `max_tokens` from 4096 → 8192. A code review (2026-06-27) identified two HIGH-severity production risks:

1. **Silent truncation** — when the LLM hits `max_tokens=8192` mid-batch, `response.stop_reason` is never inspected. Two failure modes:
   - JSON cut mid-string → `parse_translation_response` raises `"Invalid JSON at char N"`, which blames the LLM's JSON syntax instead of the real cause (truncation).
   - JSON cut at array close (valid JSON, fewer items) → `parse_translation_response` returns a short list, `translate_with_llm` accepts it, and `export_review_csv` writes empty `text_vi` fields for the missing segments — silent data loss that flows into TTS.

2. **No partial-batch recovery** — the batching loop has no retry, no checkpoint, no write-through. A transient `APIConnectionError` or HTTP 429 on batch 3 of 4 discards all 100 successfully-translated rows from batches 1–2; operator re-runs the full job at Nx cost.

These were flagged as findings #1 and #2 in the project review (2026-06-27).

## Goal

1. Detect truncation loudly with a specific error message that names the cause and the remediation.
2. Retry transient API failures per-batch with exponential backoff, propagating the error after a bounded number of attempts.

## Non-goals

- Auto-retry on truncation (truncation is structural, not transient — retrying the same batch with the same size will hit the same cap).
- Saving partial results to disk — out of scope for this fix; can be added later if operators need resume.
- Exposing retry config via `Settings` — constants are sufficient for now (YAGNI).
- Findings #3–#7 from the review (fence edge cases, model-cap validation, test coverage of batch loop, diagnostic gap) — separate specs.

## Design

### Constants

Add three module-level constants to `src/vietdub/translate.py`:

```python
LLM_MAX_RETRIES = 3           # retries after initial attempt (4 total tries)
LLM_RETRY_BACKOFF_S = 1.0     # initial backoff
LLM_RETRY_BACKOFF_FACTOR = 2.0
```

### Fix #1: truncation detection inside `_translate_one_batch`

After the LLM call and after `parse_translation_response`, add two validation checks before returning rows:

1. **`response.stop_reason == "max_tokens"`** → raise `RuntimeError`:
   ```
   MiniMax hit max_tokens ({max_tokens}) for batch of {N} segments (translated {M}). Reduce LLM_BATCH_SIZE or increase max_tokens.
   ```

2. **`len(rows) != len(segments)`** → raise `RuntimeError`:
   ```
   MiniMax returned {M} rows for batch of {N} segments; count mismatch (likely truncation). Reduce LLM_BATCH_SIZE.
   ```

Both checks fire before `_translate_one_batch` returns. The RuntimeErrors raised here are structural (not transient) and are **not** caught by the retry wrapper below.

### Fix #2: retry helper `_translate_one_batch_with_retry`

**Step 1 — let API errors propagate unwrapped from `_translate_one_batch`.**

The current `_translate_one_batch` catches `anthropic.AuthenticationError`, `anthropic.APIStatusError`, `anthropic.APIConnectionError` and re-raises each as a `RuntimeError`. For the retry wrapper to inspect the underlying exception type and `status_code`, those must propagate unwrapped. Change `_translate_one_batch`'s three `except` clauses to bare `raise` (preserving the `from exc` chain via implicit chaining).

The retry helper (below) will be responsible for wrapping non-retryable API errors with the same user-facing messages the old code produced, after the retry decision is made.

The truncation RuntimeErrors raised in Fix #1 are still raised as `RuntimeError` and are NOT caught by the retry wrapper (which only catches `anthropic.*`).

**Step 2 — add `_translate_one_batch_with_retry`:**

```python
import time

def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return _translate_one_batch(segments, context_bundle, settings)
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            # Always retryable; sleep or give up.
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
        except anthropic.APIStatusError as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in {429, 500, 502, 503, 504}:
                # Non-retryable: wrap with the same message the old code produced and re-raise.
                if isinstance(exc, anthropic.AuthenticationError):
                    raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
                raise RuntimeError(f"MiniMax HTTP {status}: {exc.message}") from exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))

    raise RuntimeError(
        f"MiniMax batch failed after {LLM_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc
```

**Retryable errors:**
- `anthropic.APIConnectionError` — network blip
- `anthropic.APIStatusError` with status in `{429, 500, 502, 503, 504}` — rate-limit or transient server error

**Non-retryable errors** (wrapped with user-facing message and re-raised, no retry):
- `anthropic.AuthenticationError` (subclass of `APIStatusError`, status 401)
- `anthropic.APIStatusError` with other 4xx statuses (400, 403, 404, etc.)
- `RuntimeError` from truncation detection (Fix #1) — structural, not caught by retry wrapper

**User-facing error message contract** (preserved from current code):
- `anthropic.AuthenticationError` → `RuntimeError("MiniMax authentication failed: {exc}")`
- `anthropic.APIStatusError` (non-retryable) → `RuntimeError("MiniMax HTTP {status_code}: {exc.message}")`
- After max retries on retryable errors → `RuntimeError("MiniMax batch failed after {N} attempts: {last_exc}")`

### Caller change

In `translate_with_llm`, swap `_translate_one_batch` → `_translate_one_batch_with_retry`:

```python
all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
```

The retry helper is the only one that imports `anthropic` indirectly via `_translate_one_batch`. The `import anthropic` inside `_translate_one_batch` stays.

### Behavior change matrix

| Failure mode | Old behavior | New behavior |
|---|---|---|
| LLM hits `max_tokens` (JSON cut mid-string) | `RuntimeError("invalid JSON at char N")` — misleading | `RuntimeError("MiniMax hit max_tokens... Reduce LLM_BATCH_SIZE or increase max_tokens")` |
| LLM hits `max_tokens` (JSON cut at array close, valid JSON, fewer items) | Silent: empty `text_vi` rows in CSV | `RuntimeError("returned M rows for batch of N segments; count mismatch")` |
| LLM returns MORE rows than input (count mismatch in either direction) | Silently dropped by `export_review_csv`'s `by_id` dict | `RuntimeError("returned M rows for batch of N segments; count mismatch")` |
| `APIConnectionError` on batch N | `RuntimeError("MiniMax network error: {exc}")`; all prior batches discarded | Retry up to 3 times with 1s/2s/4s backoff; success or `RuntimeError("MiniMax batch failed after 4 attempts: {exc}")` |
| HTTP 429 on batch N | `RuntimeError("MiniMax HTTP 429: ...")`; all prior batches discarded | Retry up to 3 times with backoff; success or wrap with retry-exhausted message |
| HTTP 500/502/503/504 on batch N | `RuntimeError("MiniMax HTTP {status}: ...")`; discarded | Retry up to 3 times with backoff |
| HTTP 400/403/404 | `RuntimeError("MiniMax HTTP {status}: ...")`; discarded | Same message, no retry (preserved) |
| `AuthenticationError` (status 401) | `RuntimeError("MiniMax authentication failed: {exc}")` | Same message, no retry (preserved) |
| Truncation after 4 retries | n/a | Same as single attempt — truncation is not retried |
| All batches succeed | Success | Success (no behavior change on happy path) |

## Test plan

Add five tests to `tests/test_translate_responses.py` (consistent with existing LLM-call tests there):

1. **`test_translate_one_batch_detects_max_tokens_truncation`**
   - Mock `_FakeAnthropic` to return a response with `stop_reason="max_tokens"` (valid JSON, full row count).
   - Call `_translate_one_batch` with 3 segments.
   - Assert: `RuntimeError` raised; message contains `"max_tokens"` and `"3"`; message contains `"Reduce LLM_BATCH_SIZE"`.

2. **`test_translate_one_batch_detects_row_count_mismatch`**
   - Mock `_FakeAnthropic` to return valid JSON with only 2 rows when batch had 3 segments. Set `stop_reason="end_turn"` (not max_tokens, so the second check is what fires).
   - Assert: `RuntimeError` raised; message contains `"2"` and `"3"`; message contains `"count mismatch"`.

3. **`test_translate_one_batch_with_retry_recovers_from_transient_connection_error`**
   - Mock `client.messages.create` to raise `anthropic.APIConnectionError` on the first 2 calls, succeed on the 3rd.
   - Call `_translate_one_batch_with_retry`.
   - Assert: returns rows; mock called 3 times; `time.sleep` called 2 times with delays matching `1.0` and `2.0` (use `monkeypatch.setattr(time, "sleep", ...)` to capture without sleeping).

4. **`test_translate_one_batch_with_retry_gives_up_after_max_attempts`**
   - Mock `client.messages.create` to always raise `anthropic.APIConnectionError`.
   - Call `_translate_one_batch_with_retry`.
   - Assert: `RuntimeError` raised; message contains `"4 attempts"` (1 initial + 3 retries); underlying exception chained via `from`.

5. **`test_translate_one_batch_with_retry_does_not_retry_non_retryable_status`**
   - Mock `client.messages.create` to raise `anthropic.APIStatusError` with status 401 (use the SDK's constructor or a stub) on every call.
   - Call `_translate_one_batch_with_retry`.
   - Assert: `RuntimeError` raised; message contains `"authentication failed"` (the preserved user-facing message); mock called exactly 1 time (no retry); `time.sleep` not called.

Existing tests in `tests/test_translate_responses.py` use single-segment inputs that don't trigger truncation or retry paths — they continue to pass unchanged. The existing `_FakeAnthropic` infrastructure (already in the file) needs to support raising `anthropic.APIConnectionError` and `anthropic.APIStatusError` for tests 3–5; extend it minimally if needed.

## Risk

**Low–Medium.** Changes are localized to `_translate_one_batch` and its caller, plus a new wrapper function. No changes to public API surface (`translate_with_llm`, `parse_translation_response` signatures unchanged). Test coverage expanded to cover both failure modes. Operators on jobs that previously hit silent truncation will now see clear errors — a visible change but the correct one.

## Rollback

Three-file change (one source file, one test file, one design doc revert). Revert the implementation commit if CI fails or operators report regressions.

## Implementation notes

- `import time` goes at the top of `translate.py` with the other stdlib imports.
- `import anthropic` stays inside `_translate_one_batch` (preserves existing pattern + works with `monkeypatch.setitem(sys.modules, ...)` in tests).
- `last_exc: Exception | None` requires `from __future__ import annotations` (already present at line 1).
- The three `except anthropic.*` clauses in `_translate_one_batch` change from `raise RuntimeError(...) from exc` to bare `raise`. The retry helper now owns the wrapping for non-retryable errors.
- Retry config is hardcoded constants, not in `Settings` — YAGNI. If operators later need to tune, move to `Settings.llm_max_retries` etc.
- No new module dependencies.
- No changes to `export_review_csv` or `parse_translation_response`.
