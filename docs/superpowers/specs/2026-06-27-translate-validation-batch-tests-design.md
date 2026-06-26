# Translate Validation + Batch-Loop Tests — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (validation) + `tests/test_translate_responses.py` (batch-loop tests)
**Status:** Approved (design), pending implementation

## Context

The 2026-06-27 review surfaced 7 findings. Specs 1 and 2 (commits `44e07a8`–`11b95a8` and `cc22910`–`2c22459`) addressed findings #1, #2, #3, #4, #7. This spec addresses the remaining two:

- **Finding #5 (max_tokens validation against model output cap):** `LLM_MAX_TOKENS = 8192` is hardcoded (`src/vietdub/translate.py:96`). If a user configures `LLM_MODEL=claude-3-haiku-20240307` (max output 4096 tokens), the API truncates responses mid-JSON. While the truncation detection from Spec 1 catches this at runtime with a clear error, failing fast at startup prevents wasted work on jobs that will inevitably fail.

- **Finding #6 (zero batch-loop test coverage):** The batching loop in `translate_with_llm` (`src/vietdub/translate.py:119-122` — `for start in range(0, len(segments), LLM_BATCH_SIZE)`) is untested. All existing tests pass 1-segment inputs (single batch). A regression in the slice, batch count, or order would not be caught.

## Goal

1. Validate `settings.llm_model` against a known-output-cap dict at `translate_with_llm` startup; raise clear error if cap is too small, warn if model is unknown.
2. Add 4 batch-loop tests covering 50 / 51 / 100 / 0 segments.

## Non-goals

- Auto-detect model output caps (no Anthropic SDK query API for this).
- Validate other settings (anthropic_base_url format, etc.) — separate spec if needed.
- Add a `--no-cap-validation` flag — YAGNI.
- Auto-retry truncation with smaller batches — already designed out per Spec 1.
- Findings #1–#4, #7 — already addressed.

## Design

### Section 1: `KNOWN_MODEL_OUTPUT_CAPS` dict

Add a module-level dict at the top of `src/vietdub/translate.py` (near the existing constants):

```python
KNOWN_MODEL_OUTPUT_CAPS: dict[str, int] = {
    "MiniMax-M3": 8192,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-sonnet-20240620": 8192,
    "claude-3-opus-20240229": 4096,
}
```

The dict lists models the project explicitly supports. Operators adding a new model append to this dict (or accept the warning). Defaults are sourced from Anthropic's published model docs as of 2026-06.

### Section 2: Validation logic in `translate_with_llm`

Add validation after the existing `llm_model` check (currently `src/vietdub/translate.py:115-116`):

```python
def translate_with_llm(segments, context_bundle, *, settings):
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for translation")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is required for translation")

    cap = KNOWN_MODEL_OUTPUT_CAPS.get(settings.llm_model)
    if cap is not None and LLM_MAX_TOKENS > cap:
        raise RuntimeError(
            f"LLM_MAX_TOKENS ({LLM_MAX_TOKENS}) exceeds known output cap "
            f"({cap}) for model {settings.llm_model!r}. "
            f"Reduce LLM_MAX_TOKENS or use a different model."
        )
    if cap is None:
        print(
            f"Warning: model {settings.llm_model!r} not in KNOWN_MODEL_OUTPUT_CAPS; "
            f"skipping output cap validation. If you hit truncation, "
            f"add the model's output cap to the dict.",
            file=sys.stderr,
        )

    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_MAX_TOKENS]  # unchanged
        all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
    return all_rows
```

**Behavior matrix:**

| Model in dict | Cap vs LLM_MAX_TOKENS | Result |
|---|---|---|
| Yes | Cap >= LLM_MAX_TOKENS | Proceed silently |
| Yes | Cap < LLM_MAX_TOKENS | Raise `RuntimeError` with model + cap |
| No | (n/a) | Warn to stderr, proceed |

### Section 3: Tests

#### Section 3a: Smart mock for batch tests

The current `_FakeMessages.create` returns the same 1-row response regardless of input. For batch tests, we need a mock that returns N rows matching the N segments in the input batch.

Add a new helper class `tests/test_translate_responses.py` (near the existing `_FakeMessages`):

```python
class _BatchAwareMessages:
    """Mock that returns rows matching the input batch size and segment IDs."""

    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Parse segment count from user_prompt (JSON-encoded in messages[0]["content"])
        user_prompt = kwargs["messages"][0]["content"]
        payload = json.loads(user_prompt)
        batch_segments = payload["segments"]
        translations = [
            {
                "segment_id": seg["id"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "speaker": seg.get("speaker"),
                "text_cn": seg.get("text", ""),
                "text_vi": f"translation of {seg['id']}",
                "context_note": "",
                "status": "draft",
            }
            for seg in batch_segments
        ]
        return types.SimpleNamespace(
            content=[_TextBlock(json.dumps({"translations": translations}))],
            stop_reason="end_turn",
        )


class _BatchAwareAnthropic:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = _BatchAwareMessages()
        self.__class__.instances.append(self)
```

Existing tests continue to use the original `_FakeAnthropic` / `_FakeMessages`. New batch tests use `_BatchAwareAnthropic`.

#### Section 3b: 7 new tests

**#5 Validation tests** (3 tests):

1. **`test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap`**
   ```python
   def test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap(monkeypatch):
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))
       with pytest.raises(RuntimeError, match="exceeds known output cap") as exc_info:
           translate_with_llm(
               segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
               context_bundle={},
               settings=_settings(llm_model="claude-3-haiku-20240307"),
           )
       assert "4096" in str(exc_info.value)
       assert "8192" in str(exc_info.value)
   ```

2. **`test_translate_with_llm_passes_when_model_cap_sufficient`**
   ```python
   def test_translate_with_llm_passes_when_model_cap_sufficient(monkeypatch):
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))
       # claude-3-5-sonnet cap=8192, LLM_MAX_TOKENS=8192 → OK
       rows = translate_with_llm(
           segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
           context_bundle={},
           settings=_settings(llm_model="claude-3-5-sonnet-20240620"),
       )
       assert len(rows) == 1
   ```

3. **`test_translate_with_llm_warns_for_unknown_model`** (uses `capsys`)
   ```python
   def test_translate_with_llm_warns_for_unknown_model(monkeypatch, capsys):
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))
       rows = translate_with_llm(
           segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
           context_bundle={},
           settings=_settings(llm_model="some-future-model-xyz"),
       )
       assert len(rows) == 1
       captured = capsys.readouterr()
       assert "some-future-model-xyz" in captured.err
       assert "KNOWN_MODEL_OUTPUT_CAPS" in captured.err
   ```

**#6 Batch-loop tests** (4 tests):

4. **`test_translate_with_llm_batches_50_segments_into_one_call`**
   ```python
   def test_translate_with_llm_batches_50_segments_into_one_call(monkeypatch):
       _BatchAwareAnthropic.instances = []
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))
       segments = [
           TimedSegment(id=f"m-{i:04d}", start_ms=i*1000, end_ms=(i+1)*1000, text=f"text {i}")
           for i in range(50)
       ]
       rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())
       assert len(rows) == 50
       assert len(_BatchAwareAnthropic.instances) == 1
       assert len(_BatchAwareAnthropic.instances[0].messages.calls) == 1
   ```

5. **`test_translate_with_llm_batches_51_segments_into_two_calls`**
   ```python
   def test_translate_with_llm_batches_51_segments_into_two_calls(monkeypatch):
       _BatchAwareAnthropic.instances = []
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))
       segments = [
           TimedSegment(id=f"m-{i:04d}", start_ms=i*1000, end_ms=(i+1)*1000, text=f"text {i}")
           for i in range(51)
       ]
       rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())
       assert len(rows) == 51
       assert len(_BatchAwareAnthropic.instances) == 2
       # First call covers m-0000..m-0049 (50 segments)
       first_payload = json.loads(_BatchAwareAnthropic.instances[0].messages.calls[0]["messages"][0]["content"])
       assert len(first_payload["segments"]) == 50
       assert first_payload["segments"][0]["id"] == "m-0000"
       # Second call covers m-0050 (1 segment)
       second_payload = json.loads(_BatchAwareAnthropic.instances[1].messages.calls[0]["messages"][0]["content"])
       assert len(second_payload["segments"]) == 1
       assert second_payload["segments"][0]["id"] == "m-0050"
   ```

6. **`test_translate_with_llm_batches_100_segments_into_two_calls`**
   ```python
   def test_translate_with_llm_batches_100_segments_into_two_calls(monkeypatch):
       _BatchAwareAnthropic.instances = []
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))
       segments = [
           TimedSegment(id=f"m-{i:04d}", start_ms=i*1000, end_ms=(i+1)*1000, text=f"text {i}")
           for i in range(100)
       ]
       rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())
       assert len(rows) == 100
       assert len(_BatchAwareAnthropic.instances) == 2
       for instance in _BatchAwareAnthropic.instances:
           assert len(instance.messages.calls[0]["messages"]) > 0
   ```

7. **`test_translate_with_llm_handles_empty_segments`**
   ```python
   def test_translate_with_llm_handles_empty_segments(monkeypatch):
       _BatchAwareAnthropic.instances = []
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))
       rows = translate_with_llm(segments=[], context_bundle={}, settings=_settings())
       assert rows == []
       assert len(_BatchAwareAnthropic.instances) == 0
   ```

### Section 4: Implementation notes

- `KNOWN_MODEL_OUTPUT_CAPS` is a module-level constant dict (not in `Settings`). Operators edit the source to add models. YAGNI for runtime configurability.
- Validation runs at the start of `translate_with_llm`, before any LLM call. Failed validation → no API call attempted → no cost.
- The warning for unknown models goes to `sys.stderr` (matching the existing `print(..., file=sys.stderr)` pattern in `parse_translation_response` for status coercion).
- `_BatchAwareAnthropic.instances` is a class-level list shared across tests. Each batch test must reset it (`_BatchAwareAnthropic.instances = []`) before running to avoid pollution from prior tests.
- The 7 new tests use the existing `_settings()` helper (no new settings fields needed — `llm_model` is the only one overridden).
- No changes to `_translate_one_batch`, `_translate_one_batch_with_retry`, retry constants, `_strip_markdown_fences`, or `parse_translation_response`.

## Risk

**Low.** Validation is a single new check at function entry; it doesn't change the existing call paths. Batch-loop tests are additive — they exercise existing code paths with new inputs.

## Rollback

Single source file change + tests. Revert the implementation commit if CI fails or operators report regressions.

## Implementation notes (summary)

- Add `KNOWN_MODEL_OUTPUT_CAPS` dict near `LLM_MAX_TOKENS` constant (around line 96-99).
- Add validation block after `if not settings.llm_model:` check.
- Add `_BatchAwareMessages` and `_BatchAwareAnthropic` helper classes to `tests/test_translate_responses.py`.
- Add 7 new tests in `tests/test_translate_responses.py`.
- One commit for all changes.
