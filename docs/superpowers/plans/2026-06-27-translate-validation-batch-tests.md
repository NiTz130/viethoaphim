# Translate Validation + Batch-Loop Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate `settings.llm_model` against a known-output-cap dict at `translate_with_llm` startup, and add batch-loop tests covering 50/51/100/0 segments.

**Architecture:** Add a module-level `KNOWN_MODEL_OUTPUT_CAPS` dict + validation block in `translate_with_llm`. Add a `_BatchAwareAnthropic` mock that returns rows matching the input batch size, then add 4 batch-loop tests that exercise the `for start in range(0, len(segments), LLM_BATCH_SIZE)` loop with various segment counts.

**Tech Stack:** Python 3.11+, pytest, monkeypatch.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-translate-validation-batch-tests-design.md` (authoritative for behavior).
- `KNOWN_MODEL_OUTPUT_CAPS` dict with 4 models:
  - `"MiniMax-M3": 8192`
  - `"claude-3-haiku-20240307": 4096`
  - `"claude-3-5-sonnet-20240620": 8192`
  - `"claude-3-opus-20240229": 4096`
- Validation rules:
  - Model in dict + cap < `LLM_MAX_TOKENS` → raise `RuntimeError` with model name + caps
  - Model in dict + cap >= `LLM_MAX_TOKENS` → proceed silently
  - Model not in dict → print warning to `sys.stderr`, proceed
- `LLM_MAX_TOKENS = 8192` (existing constant).
- TDD discipline: write tests first, run to verify, implement, run to verify pass.
- Existing tests in `tests/test_translate_responses.py` must continue to pass.
- `_BatchAwareAnthropic.instances` is class-level and must be reset to `[]` at the start of each batch test (avoids pollution from prior tests).
- No changes to `_translate_one_batch`, `_translate_one_batch_with_retry`, retry constants, `_strip_markdown_fences`, `parse_translation_response`, or `Settings`.

---

### Task 1: Add `KNOWN_MODEL_OUTPUT_CAPS` dict and validation logic

**Files:**
- Modify: `src/vietdub/translate.py` (add dict near existing constants around line 96-99; add validation block after the `llm_model` check around line 116)
- Modify: `tests/test_translate_responses.py` (add 3 tests at the end)

**Interfaces:**
- Consumes: `settings.llm_model` (str).
- Produces: raises `RuntimeError` or prints to `sys.stderr`.

- [ ] **Step 1: Add 3 failing tests**

Add the following to `tests/test_translate_responses.py` at the end of the file:

```python
def test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    with pytest.raises(RuntimeError, match="exceeds known output cap") as exc_info:
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(llm_model="claude-3-haiku-20240307"),
        )

    msg = str(exc_info.value)
    assert "4096" in msg
    assert "8192" in msg


def test_translate_with_llm_passes_when_model_cap_sufficient(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
        context_bundle={},
        settings=_settings(llm_model="claude-3-5-sonnet-20240620"),
    )

    assert len(rows) == 1


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

- [ ] **Step 2: Run new tests to verify expected failures**

Run: `pytest tests/test_translate_responses.py::test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap tests/test_translate_responses.py::test_translate_with_llm_passes_when_model_cap_sufficient tests/test_translate_responses.py::test_translate_with_llm_warns_for_unknown_model -v`

Expected:
- `test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap`: **PASS** (no validation = no error, so `pytest.raises(RuntimeError, ...)` finds no match → fails)

Wait — `pytest.raises(RuntimeError)` requires that a RuntimeError IS raised. If no exception is raised, `pytest.raises` fails the test. So this test FAILS at RED because no RuntimeError is raised.

- `test_translate_with_llm_passes_when_model_cap_sufficient`: **PASS** (no validation = no error, translation succeeds, returns 1 row → assertions match)
- `test_translate_with_llm_warns_for_unknown_model`: **FAIL** (no warning emitted, `capsys.readouterr().err` empty → assertion `"some-future-model-xyz" in captured.err` fails)

So expected at RED: 2 fail (the "raises" and "warns" tests), 1 pass (the "passes" test).

- [ ] **Step 3: Add `KNOWN_MODEL_OUTPUT_CAPS` dict**

In `src/vietdub/translate.py`, after the existing `LLM_RETRY_BACKOFF_FACTOR = 2.0` line (currently line 99), add:

```python
KNOWN_MODEL_OUTPUT_CAPS: dict[str, int] = {
    "MiniMax-M3": 8192,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-sonnet-20240620": 8192,
    "claude-3-opus-20240229": 4096,
}
```

- [ ] **Step 4: Add validation block in `translate_with_llm`**

In `src/vietdub/translate.py`, after the existing `if not settings.llm_model:` check (currently lines 115-116), add:

```python
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
```

The complete `translate_with_llm` head should now look like:

```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
) -> list[TranslationRow]:
    """Translate segments by batching them into LLM calls.

    A single LLM call cannot handle hundreds of segments within max_tokens
    limits, so we batch into groups of LLM_BATCH_SIZE and concatenate results.
    """
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
        batch = segments[start:start + LLM_BATCH_SIZE]
        all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
    return all_rows
```

- [ ] **Step 5: Run new tests to verify all pass**

Run: `pytest tests/test_translate_responses.py::test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap tests/test_translate_responses.py::test_translate_with_llm_passes_when_model_cap_sufficient tests/test_translate_responses.py::test_translate_with_llm_warns_for_unknown_model -v`

Expected: all 3 pass.

- [ ] **Step 6: Run full suite to check for regressions**

Run: `pytest -v`

Expected: 121/121 pass (was 118 + 3 new = 121). Existing tests use `_settings()` default (`llm_model="MiniMax-M3"`) which is in `KNOWN_MODEL_OUTPUT_CAPS` with cap 8192 == `LLM_MAX_TOKENS`, so they pass without warnings.

- [ ] **Step 7: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate_responses.py
git commit -m "feat(translate): validate llm_model output cap at startup

Add KNOWN_MODEL_OUTPUT_CAPS dict mapping model name to max output tokens.
Raise clear RuntimeError when LLM_MAX_TOKENS exceeds a known model's cap.
Warn to stderr for unknown models so operators can update the dict."
```

---

### Task 2: Add `_BatchAwareAnthropic` mock + batch-loop tests

**Files:**
- Modify: `tests/test_translate_responses.py` (add mock classes + 4 tests at the end)

**Interfaces:**
- Consumes: existing `translate_with_llm`, existing `_TextBlock`, existing `_settings()`.
- Produces: 4 new tests that exercise the batching loop with various segment counts.

- [ ] **Step 1: Add `_BatchAwareMessages` and `_BatchAwareAnthropic` helper classes**

Add to `tests/test_translate_responses.py` immediately after the existing `_FakeAnthropic` class (around line 45, before `_settings`):

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

- [ ] **Step 2: Add 4 batch-loop tests**

Add to `tests/test_translate_responses.py` at the end of the file:

```python
def test_translate_with_llm_batches_50_segments_into_one_call(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(50)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 50
    assert len(_BatchAwareAnthropic.instances) == 1
    assert len(_BatchAwareAnthropic.instances[0].messages.calls) == 1


def test_translate_with_llm_batches_51_segments_into_two_calls(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(51)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 51
    assert len(_BatchAwareAnthropic.instances) == 2
    # First call covers m-0000..m-0049 (50 segments)
    first_payload = json.loads(
        _BatchAwareAnthropic.instances[0].messages.calls[0]["messages"][0]["content"]
    )
    assert len(first_payload["segments"]) == 50
    assert first_payload["segments"][0]["id"] == "m-0000"
    # Second call covers m-0050 (1 segment)
    second_payload = json.loads(
        _BatchAwareAnthropic.instances[1].messages.calls[0]["messages"][0]["content"]
    )
    assert len(second_payload["segments"]) == 1
    assert second_payload["segments"][0]["id"] == "m-0050"


def test_translate_with_llm_batches_100_segments_into_two_calls(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(100)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 100
    assert len(_BatchAwareAnthropic.instances) == 2


def test_translate_with_llm_handles_empty_segments(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    rows = translate_with_llm(segments=[], context_bundle={}, settings=_settings())

    assert rows == []
    assert len(_BatchAwareAnthropic.instances) == 0
```

- [ ] **Step 3: Run new tests to verify all pass**

Run: `pytest tests/test_translate_responses.py::test_translate_with_llm_batches_50_segments_into_one_call tests/test_translate_responses.py::test_translate_with_llm_batches_51_segments_into_two_calls tests/test_translate_responses.py::test_translate_with_llm_batches_100_segments_into_two_calls tests/test_translate_responses.py::test_translate_with_llm_handles_empty_segments -v`

Expected: all 4 pass. (These are behavior-spec tests that exercise existing batching logic — no TDD red expected because the code paths already work correctly.)

- [ ] **Step 4: Run full suite to verify no regressions**

Run: `pytest -v`

Expected: 125/125 pass (was 121 + 4 new = 125).

- [ ] **Step 5: Commit**

```bash
git add tests/test_translate_responses.py
git commit -m "test(translate): cover batching loop with 50/51/100/0 segments

Add _BatchAwareAnthropic mock that returns rows matching the input
batch size, then exercise the for-loop in translate_with_llm with
4 new tests covering boundary (50 = 1 batch), off-by-one (51 = 2
batches), full batches (100 = 2 batches), and empty input (0 = 0
batches)."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| `KNOWN_MODEL_OUTPUT_CAPS` dict (4 models) | Task 1 Step 3 |
| Validation raises on cap < LLM_MAX_TOKENS | Task 1 Step 4 |
| Validation warns on unknown model | Task 1 Step 4 |
| Test raises when cap exceeded | Task 1 Step 1 |
| Test passes when cap sufficient | Task 1 Step 1 |
| Test warns for unknown model | Task 1 Step 1 |
| `_BatchAwareMessages` mock | Task 2 Step 1 |
| `_BatchAwareAnthropic` class | Task 2 Step 1 |
| Test 50 segments → 1 batch | Task 2 Step 2 |
| Test 51 segments → 2 batches | Task 2 Step 2 |
| Test 100 segments → 2 batches | Task 2 Step 2 |
| Test empty segments → 0 batches | Task 2 Step 2 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later", "fill in details".
- No "Add appropriate error handling" without specific code.
- Every code change shows full code.
- Every test has full assertions.
- Every command has expected output.

**3. Type consistency:**

- `KNOWN_MODEL_OUTPUT_CAPS: dict[str, int]` — matches everywhere.
- `_BatchAwareAnthropic.instances` — class-level, reset at start of each test.
- `translate_with_llm` signature unchanged.
- `_BatchAwareMessages.create(**kwargs)` signature matches Anthropic's `messages.create`.

**4. Test/impl ordering:**

- Task 1 follows strict TDD: 3 tests, RED verification, implementation, GREEN verification.
- Task 2 adds tests for existing behavior (no implementation change), so all tests pass immediately at GREEN. This is acceptable — these are behavior-spec tests that lock in existing correct behavior and guard against future regressions.
- Both tasks end with full suite green before commit.
