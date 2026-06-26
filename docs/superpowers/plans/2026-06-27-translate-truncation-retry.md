# Translate Truncation Detection + Per-Batch Retry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect LLM truncation loudly with clear errors and retry transient API failures per-batch with exponential backoff.

**Architecture:** Add two checks in `_translate_one_batch` (`response.stop_reason == "max_tokens"` and `len(rows) != len(segments)`) that raise clear `RuntimeError`s on truncation. Wrap that function in a new `_translate_one_batch_with_retry` helper that catches transient `anthropic.*` errors (network, 429, 5xx) and retries with `1s/2s/4s` backoff up to 3 retries. Truncation `RuntimeError`s propagate unchanged (structural, not transient).

**Tech Stack:** Python 3.x, anthropic SDK, pytest, monkeypatch.

## Global Constraints

- Backoff constants: `LLM_MAX_RETRIES = 3`, `LLM_RETRY_BACKOFF_S = 1.0`, `LLM_RETRY_BACKOFF_FACTOR = 2.0` (module-level in `src/vietdub/translate.py`).
- Retryable HTTP statuses: `{429, 500, 502, 503, 504}`. `anthropic.APIConnectionError` always retryable. Other `APIStatusError` and `AuthenticationError` not retryable.
- Total attempts: `1 + LLM_MAX_RETRIES = 4`. Final error message after exhaustion: `"MiniMax batch failed after 4 attempts: {last_exc}"`.
- TDD discipline: write failing test, run to verify failure, implement minimal code, run to verify pass. Each task ends with a green test run.
- `from __future__ import annotations` already present in `translate.py` (line 1) — `Exception | None` and `list[...]` syntax works without `typing.Optional` / `List`.
- Existing tests in `tests/test_translate_responses.py` that go through `translate_with_llm` with mocked `anthropic` must set `Anthropic`, `AuthenticationError`, `APIStatusError`, and `APIConnectionError` on the monkeypatched module — the helper inspects all of them.
- Spec: `docs/superpowers/specs/2026-06-27-translate-truncation-retry-design.md` (authoritative for behavior matrix and error message wording).

---

### Task 1: Add `_translate_one_batch_with_retry` and wire into `translate_with_llm`

**Files:**
- Modify: `src/vietdub/translate.py:1-7` (imports), `src/vietdub/translate.py:94` (after `LLM_BATCH_SIZE`), `src/vietdub/translate.py:113-117` (call site), `src/vietdub/translate.py` (new function after `_translate_one_batch`)
- Modify: `tests/test_translate_responses.py` (add 3 new tests)

**Interfaces:**
- Consumes: `_translate_one_batch(segments: list[TimedSegment], context_bundle: dict, settings) -> list[TranslationRow]` (existing).
- Produces: `_translate_one_batch_with_retry(segments: list[TimedSegment], context_bundle: dict, settings) -> list[TranslationRow]` (new).

- [ ] **Step 1: Write 3 failing tests**

Add the following to `tests/test_translate_responses.py` at the end of the file:

```python
def test_translate_one_batch_with_retry_recovers_from_transient_connection_error(monkeypatch):
    class _ConnError(Exception):
        pass

    call_count = [0]

    class _CountingMessages:
        def create(self, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise _ConnError()
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "你好",
                        "text_vi": "Xin chào",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _CountingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _CountingMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_CountingAnthropic,
        APIConnectionError=_ConnError,
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    ))

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    from vietdub.translate import _translate_one_batch_with_retry
    rows = _translate_one_batch_with_retry(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
        context_bundle={},
        settings=_settings(),
    )

    assert call_count[0] == 3
    assert sleep_calls == [1.0, 2.0]
    assert rows[0].text_vi == "Xin chào"


def test_translate_one_batch_with_retry_gives_up_after_max_attempts(monkeypatch):
    class _ConnError(Exception):
        pass

    class _AlwaysFailMessages:
        def create(self, **kwargs):
            raise _ConnError()

    class _AlwaysFailAnthropic:
        def __init__(self, **kwargs):
            self.messages = _AlwaysFailMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_AlwaysFailAnthropic,
        APIConnectionError=_ConnError,
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    ))

    monkeypatch.setattr("time.sleep", lambda _: None)

    from vietdub.translate import _translate_one_batch_with_retry
    with pytest.raises(RuntimeError, match="4 attempts") as exc_info:
        _translate_one_batch_with_retry(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, _ConnError)


def test_translate_one_batch_with_retry_does_not_retry_non_retryable_status(monkeypatch):
    class _AuthError(Exception):
        def __str__(self):
            return "bad key"

    class _AuthMessages:
        def create(self, **kwargs):
            raise _AuthError()

    class _AuthAnthropic:
        def __init__(self, **kwargs):
            self.messages = _AuthMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_AuthAnthropic,
        AuthenticationError=_AuthError,
        APIStatusError=_AuthError,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    from vietdub.translate import _translate_one_batch_with_retry
    with pytest.raises(RuntimeError, match="authentication failed"):
        _translate_one_batch_with_retry(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )

    # Call count check: raise from messages.create is caught once, wrapped, re-raised.
    # Verify no retries happened.
    assert sleep_calls == []
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_translate_responses.py::test_translate_one_batch_with_retry_recovers_from_transient_connection_error tests/test_translate_responses.py::test_translate_one_batch_with_retry_gives_up_after_max_attempts tests/test_translate_responses.py::test_translate_one_batch_with_retry_does_not_retry_non_retryable_status -v`

Expected: all 3 fail with `ImportError: cannot import name '_translate_one_batch_with_retry' from 'vietdub.translate'`.

- [ ] **Step 3: Add `import time` and 3 retry constants**

In `src/vietdub/translate.py`, after line 2 (`import csv`), add:
```python
import time
```

After the existing line `LLM_BATCH_SIZE = 50` (line 94), add:
```python
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_S = 1.0
LLM_RETRY_BACKOFF_FACTOR = 2.0
```

- [ ] **Step 4: Implement `_translate_one_batch_with_retry`**

Add the following function to `src/vietdub/translate.py` immediately after `_translate_one_batch` ends (after its `return parse_translation_response(content)` line):

```python
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
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
        except anthropic.APIStatusError as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in {429, 500, 502, 503, 504}:
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

Note: `anthropic` is imported lazily inside `_translate_one_batch`. When this function is called, the caller must have already triggered `_translate_one_batch` (which imports anthropic). If the retry helper is called in isolation in a test, the test's monkeypatched `anthropic` module is used.

- [ ] **Step 5: Run new tests to verify they pass**

Run: `pytest tests/test_translate_responses.py::test_translate_one_batch_with_retry_recovers_from_transient_connection_error tests/test_translate_responses.py::test_translate_one_batch_with_retry_gives_up_after_max_attempts tests/test_translate_responses.py::test_translate_one_batch_with_retry_does_not_retry_non_retryable_status -v`

Expected: all 3 pass.

- [ ] **Step 6: Wire helper into `translate_with_llm`**

In `src/vietdub/translate.py`, change the line inside `translate_with_llm`'s loop (currently line 116):
```python
all_rows.extend(_translate_one_batch(batch, context_bundle, settings))
```
to:
```python
all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/test_translate.py tests/test_translate_responses.py -v`

Expected: all tests pass. The helper is wired but `_translate_one_batch` still wraps API errors in `RuntimeError`, so the helper's `except anthropic.*` clauses never match — wrapped `RuntimeError`s propagate unchanged with the same message as before.

- [ ] **Step 8: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate_responses.py
git commit -m "feat(translate): add retry helper for transient API failures

_translate_one_batch_with_retry wraps the single-shot helper with
exponential-backoff retry on transient errors (network, 429, 5xx).
1 initial + 3 retries with 1s/2s/4s backoff. User-facing messages
for non-retryable errors preserve the existing wording."
```

---

### Task 2: Unwrap API errors in `_translate_one_batch` + update existing tests

**Files:**
- Modify: `src/vietdub/translate.py:144-149` (3 except clauses)
- Modify: `tests/test_translate_responses.py:78-156` (3 existing tests)

This task is the second half of the retry integration. After this task, the retry helper actually sees `anthropic.*` exceptions (not wrapped `RuntimeError`s) and retries can fire.

- [ ] **Step 1: Update `test_translate_with_llm_maps_authentication_error`**

The retry helper's `except anthropic.APIStatusError` clause inspects the exception type. If the test's monkeypatched `anthropic` module only sets `AuthenticationError` (without `APIStatusError`), the `except` clause raises `AttributeError` at lookup time. Add `APIStatusError` and `APIConnectionError` to the monkeypatch so the helper can resolve them.

In `tests/test_translate_responses.py`, replace the existing `test_translate_with_llm_maps_authentication_error` (lines 78-101) with:

```python
def test_translate_with_llm_maps_authentication_error(monkeypatch):
    class _AuthError(Exception):
        def __init__(self):
            super().__init__("bad key")

    class _BoomMessages:
        def create(self, **kwargs):
            raise _AuthError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        AuthenticationError=_AuthError,
        APIStatusError=_AuthError,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    with pytest.raises(RuntimeError, match="authentication failed"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )
```

Note: `AuthenticationError` and `APIStatusError` point to the same class so `isinstance(exc, AuthenticationError)` is True in the retry helper.

- [ ] **Step 2: Update `test_translate_with_llm_maps_http_status_error`**

Status 429 is now retryable. The test currently expects `"HTTP 429"` in the message; after unwrap, the message becomes `"MiniMax batch failed after 4 attempts: ..."` because all 4 attempts fail and retries are exhausted. Replace the existing `test_translate_with_llm_maps_http_status_error` (lines 104-129) with:

```python
def test_translate_with_llm_maps_http_status_error_after_retry(monkeypatch):
    class _StatusError(Exception):
        status_code = 429
        message = "rate limited"

    class _BoomMessages:
        def create(self, **kwargs):
            raise _StatusError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        APIStatusError=_StatusError,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="4 attempts"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )
```

Renamed (`_after_retry` suffix) for clarity. `time.sleep` mocked to no-op so the test runs in <1s instead of 7s.

- [ ] **Step 3: Update `test_translate_with_llm_maps_connection_error`**

`APIConnectionError` is always retryable. Same pattern as the HTTP 429 test. Replace the existing `test_translate_with_llm_maps_connection_error` (lines 132-156) with:

```python
def test_translate_with_llm_maps_connection_error_after_retry(monkeypatch):
    class _ConnError(Exception):
        pass

    class _BoomMessages:
        def create(self, **kwargs):
            raise _ConnError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        APIConnectionError=_ConnError,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
    ))

    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="4 attempts"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )
```

- [ ] **Step 4: Run updated tests to verify they fail (current wrapping still in place)**

Run: `pytest tests/test_translate_responses.py::test_translate_with_llm_maps_authentication_error tests/test_translate_responses.py::test_translate_with_llm_maps_http_status_error_after_retry tests/test_translate_responses.py::test_translate_with_llm_maps_connection_error_after_retry -v`

Expected:
- `test_translate_with_llm_maps_authentication_error`: PASSES (still works because `AuthenticationError` and `APIStatusError` are the same class, so the helper treats it as non-retryable and wraps with "authentication failed").
- `test_translate_with_llm_maps_http_status_error_after_retry`: FAILS — message contains `"HTTP 429"` (current wrapping), not `"4 attempts"` (expected after unwrap).
- `test_translate_with_llm_maps_connection_error_after_retry`: FAILS — message contains `"network error"` (current wrapping), not `"4 attempts"`.

This is the expected failing state before unwrap.

- [ ] **Step 5: Unwrap API errors in `_translate_one_batch`**

In `src/vietdub/translate.py`, replace the three `except anthropic.* as exc:` clauses (currently lines 144-149) with bare `raise`:

```python
    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIStatusError:
        raise
    except anthropic.APIConnectionError:
        raise
```

The retry helper now owns the user-facing wrapping for non-retryable API errors. The `from exc` chain is preserved implicitly by the bare `raise`.

- [ ] **Step 6: Run updated tests to verify they pass**

Run: `pytest tests/test_translate_responses.py::test_translate_with_llm_maps_authentication_error tests/test_translate_responses.py::test_translate_with_llm_maps_http_status_error_after_retry tests/test_translate_responses.py::test_translate_with_llm_maps_connection_error_after_retry -v`

Expected: all 3 pass.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/test_translate.py tests/test_translate_responses.py -v`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate_responses.py
git commit -m "refactor(translate): unwrap API errors for retry helper

Bare raise in _translate_one_batch instead of wrapping as RuntimeError.
The retry helper now owns user-facing wrapping for non-retryable errors.
Existing tests for HTTP 429 and connection errors updated to expect
retry-exhausted message after 4 attempts."
```

---

### Task 3: Add truncation detection to `_translate_one_batch` (TDD)

**Files:**
- Modify: `src/vietdub/translate.py:151-156` (after parse, before return)
- Modify: `tests/test_translate_responses.py` (add 2 new tests)

- [ ] **Step 1: Write 2 failing tests**

Add to `tests/test_translate_responses.py` at the end of the file:

```python
def test_translate_one_batch_detects_max_tokens_truncation(monkeypatch):
    class _TruncatedMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [
                        {
                            "segment_id": "m-0001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "speaker": None,
                            "text_cn": "你好",
                            "text_vi": "Xin chào",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0002",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "speaker": None,
                            "text_cn": "世界",
                            "text_vi": "Thế giới",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0003",
                            "start_ms": 2000,
                            "end_ms": 3000,
                            "speaker": None,
                            "text_cn": "朋友",
                            "text_vi": "Bạn bè",
                            "context_note": "",
                            "status": "draft",
                        },
                    ]
                }))],
                stop_reason="max_tokens",
            )

    class _TruncatedAnthropic:
        def __init__(self, **kwargs):
            self.messages = _TruncatedMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_TruncatedAnthropic,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    from vietdub.translate import _translate_one_batch
    with pytest.raises(RuntimeError, match="max_tokens") as exc_info:
        _translate_one_batch(
            segments=[
                TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好"),
                TimedSegment(id="m-0002", start_ms=1000, end_ms=2000, text="世界"),
                TimedSegment(id="m-0003", start_ms=2000, end_ms=3000, text="朋友"),
            ],
            context_bundle={},
            settings=_settings(),
        )

    msg = str(exc_info.value)
    assert "3" in msg
    assert "Reduce LLM_BATCH_SIZE" in msg


def test_translate_one_batch_detects_row_count_mismatch(monkeypatch):
    class _ShortMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [
                        {
                            "segment_id": "m-0001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "speaker": None,
                            "text_cn": "你好",
                            "text_vi": "Xin chào",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0002",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "speaker": None,
                            "text_cn": "世界",
                            "text_vi": "Thế giới",
                            "context_note": "",
                            "status": "draft",
                        },
                    ]
                }))],
                stop_reason="end_turn",
            )

    class _ShortAnthropic:
        def __init__(self, **kwargs):
            self.messages = _ShortMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_ShortAnthropic,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    from vietdub.translate import _translate_one_batch
    with pytest.raises(RuntimeError, match="count mismatch") as exc_info:
        _translate_one_batch(
            segments=[
                TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好"),
                TimedSegment(id="m-0002", start_ms=1000, end_ms=2000, text="世界"),
                TimedSegment(id="m-0003", start_ms=2000, end_ms=3000, text="朋友"),
            ],
            context_bundle={},
            settings=_settings(),
        )

    msg = str(exc_info.value)
    assert "2" in msg
    assert "3" in msg
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_translate_responses.py::test_translate_one_batch_detects_max_tokens_truncation tests/test_translate_responses.py::test_translate_one_batch_detects_row_count_mismatch -v`

Expected: both fail. The first fails because the response is parsed and returned without raising (no stop_reason check). The second fails for the same reason (no count check; the function returns the 2 rows without complaint).

- [ ] **Step 3: Add truncation detection**

In `src/vietdub/translate.py`, change the end of `_translate_one_batch` (currently):
```python
    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax returned empty response (no text content blocks)")
    content = "".join(text_parts)
    return parse_translation_response(content)
```
to:
```python
    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax returned empty response (no text content blocks)")
    content = "".join(text_parts)
    rows = parse_translation_response(content)

    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"MiniMax hit max_tokens (8192) for batch of {len(segments)} segments "
            f"(translated {len(rows)}). Reduce LLM_BATCH_SIZE or increase max_tokens."
        )
    if len(rows) != len(segments):
        raise RuntimeError(
            f"MiniMax returned {len(rows)} rows for batch of {len(segments)} segments; "
            f"count mismatch (likely truncation). Reduce LLM_BATCH_SIZE."
        )

    return rows
```

The `8192` in the message is hardcoded to match the `max_tokens=8192` argument 4 lines above. If that value is ever made configurable, update both.

- [ ] **Step 4: Run new tests to verify they pass**

Run: `pytest tests/test_translate_responses.py::test_translate_one_batch_detects_max_tokens_truncation tests/test_translate_responses.py::test_translate_one_batch_detects_row_count_mismatch -v`

Expected: both pass.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/test_translate.py tests/test_translate_responses.py -v`

Expected: all tests pass. The existing happy-path tests (`test_translate_with_llm_calls_anthropic_messages_endpoint`, etc.) use mocks that don't set `stop_reason`, so the new check is skipped (`getattr` returns None, which is not `"max_tokens"`), and `len(rows) == len(segments)` for single-segment inputs.

- [ ] **Step 6: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate_responses.py
git commit -m "feat(translate): detect LLM truncation in _translate_one_batch

Two checks before returning rows:
- response.stop_reason == 'max_tokens' → clear error naming cause
- len(rows) != len(segments) → catches valid-JSON-but-short responses
  that previously leaked through as silent empty text_vi in CSV."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| 3 module constants | Task 1 |
| `import time` | Task 1 |
| `_translate_one_batch_with_retry` helper | Task 1 |
| Wire helper into `translate_with_llm` | Task 1 |
| Retry on `APIConnectionError` | Task 1 |
| Retry on `APIStatusError` with retryable status | Task 1 |
| Non-retryable error wrapping with preserved messages | Task 1, Task 2 |
| Bare `raise` in `_translate_one_batch` except clauses | Task 2 |
| Update existing tests for new retry behavior | Task 2 |
| `stop_reason == "max_tokens"` check | Task 3 |
| `len(rows) != len(segments)` check | Task 3 |
| Truncation RuntimeError messages with remediation hint | Task 3 |
| 5 new tests | Task 1 (3) + Task 3 (2) |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later", "fill in details".
- No "Add appropriate error handling" without specific code.
- No "Similar to Task N" without repeated code.
- Every step with code change has the actual code.
- Every test has full assertions, not "assert it works".
- Every command has expected output.

**3. Type consistency:**

- `_translate_one_batch(segments, context_bundle, settings) -> list[TranslationRow]` — used consistently in Tasks 1 and 3.
- `_translate_one_batch_with_retry(segments, context_bundle, settings) -> list[TranslationRow]` — used consistently in Task 1 and Task 2.
- `LLM_MAX_RETRIES`, `LLM_RETRY_BACKOFF_S`, `LLM_RETRY_BACKOFF_FACTOR` — used consistently.
- Retryable set `{429, 500, 502, 503, 504}` — same in spec, retry helper code, and tests.

No mismatches.

**4. Test/impl ordering:**

Each task's test step runs before the implementation step. Intermediate commits (after Task 1 Step 7, after Task 2 Step 7, after Task 3 Step 5) all leave the test suite green. Task 2 deliberately has a failing test step (Step 4) before the unwrap step (Step 5) — this is the expected TDD red state, resolved within the task before commit.
