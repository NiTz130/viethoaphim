# Translation Multi-Pass Review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in length review pass that refines translations to better match `target_vi_chars`, reducing the noisy `[LEN WARNING]` stderr output from Spec 2.

**Architecture:** Single source-file change to `src/vietdub/translate.py` (add `_review_batch_for_length` function + `review` parameter on 2 functions) + small change to `src/vietdub/cli.py` (add `--no-review` flag) + 5 tests in `tests/test_translate.py`. The review function makes a second LLM call per batch to refine translations, using `parse_translation_response` to parse the result and `row.model_copy(update={...})` to apply refined text_vi while preserving all other row fields.

**Tech Stack:** Python 3.11+, pytest, `monkeypatch`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-multi-pass-review-design.md` (authoritative).
- `_review_batch_for_length(rows, context_bundle, settings) -> list[TranslationRow]` is a new private helper.
- `review: bool = True` is a new parameter on `_translate_one_batch_with_retry` and `translate_with_llm`.
- CLI gets `--no-review: bool = False` flag (default False = review enabled).
- LLM call inside review function is NOT retried separately — outer `_translate_one_batch_with_retry` retries the whole cycle.
- Review uses `parse_translation_response` for output parsing (reuses Spec 1's fence handling + Spec 2's add_note error context).
- Refined text_vi applied via `row.model_copy(update={"text_vi": new_text})` (preserves all other fields).
- Skip conditions: empty refined text_vi → keep original; missing segment_id in response → keep original.
- Cost: 2x LLM calls per batch when review=True (default). 1x when review=False.
- Existing 140 tests must continue to pass — `_FakeAnthropic` mocks will need updating for tests that count LLM calls (existing mocks return 1 row per call; review pass will call again with same mock → may need 2 distinct mocks).

---

### Task 1: Add review pass + wire + CLI flag + 5 tests

**Files:**
- Modify: `src/vietdub/translate.py` (add `_review_batch_for_length` helper; add `review: bool = True` to `_translate_one_batch_with_retry` and `translate_with_llm`)
- Modify: `src/vietdub/cli.py` (add `--no-review` flag; pass `review=not no_review` to `run_review_pipeline`)
- Modify: `tests/test_translate.py` (add 5 tests at the end)

**Interfaces:**
- Consumes: existing `TranslationRow`, `parse_translation_response`, `_length_target_vi_chars`, `_translate_one_batch`, `Settings`.
- Produces: `translate_with_llm` returns same `TranslationRow[]` (interface unchanged), but internally runs an extra LLM call per batch when `review=True`. CLI accepts `--no-review` flag.

- [ ] **Step 1: Add 5 failing tests**

Add the following to `tests/test_translate.py` at the end of the file:

```python
def test_review_batch_for_length_returns_refined_translations(monkeypatch):
    """Review pass returns refined translations that match length target."""
    from vietdub.translate import _review_batch_for_length

    # Build initial rows (oversized)
    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好世界", text_vi="Xin chào bạn ơi nhé nhé nhé",
            context_note="", status="draft",
        )
    ]

    # Mock anthropic to return a shorter text_vi
    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _RefiningMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "text_vi": "Xin chào bạn",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _RefiningAnthropic:
        def __init__(self, **kwargs):
            self.messages = _RefiningMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_RefiningAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert len(refined) == 1
    assert refined[0].text_vi == "Xin chào bạn"
    # Original fields preserved
    assert refined[0].segment_id == "m-0001"
    assert refined[0].start_ms == 0
    assert refined[0].end_ms == 1000


def test_review_batch_for_length_preserves_unchanged_translations(monkeypatch):
    """Review pass preserves translations already within budget."""
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好", text_vi="Xin chào",  # short, within budget
            context_note="", status="draft",
        )
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns SAME text (review should preserve it)
    class _PreservingMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "text_vi": "Xin chào",  # unchanged
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _PreservingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _PreservingMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_PreservingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "Xin chào"


def test_review_batch_for_length_handles_partial_review_response(monkeypatch):
    """If review returns only some segments, keep originals for missing ones."""
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="a", text_vi="orig1", context_note="", status="draft"),
        TranslationRow(segment_id="m-0002", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="b", text_vi="orig2", context_note="", status="draft"),
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns only 1 of 2 segments
    class _PartialMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "text_vi": "refined1",
                    }]  # m-0002 missing
                }))],
                stop_reason="end_turn",
            )

    class _PartialAnthropic:
        def __init__(self, **kwargs):
            self.messages = _PartialMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_PartialAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "refined1"  # updated
    assert refined[1].text_vi == "orig2"     # kept original


def test_review_batch_for_length_skips_empty_refinement(monkeypatch):
    """If review returns empty text_vi, keep original."""
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="a", text_vi="original text", context_note="", status="draft"),
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns EMPTY text_vi
    class _EmptyMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "text_vi": "",  # empty!
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _EmptyAnthropic:
        def __init__(self, **kwargs):
            self.messages = _EmptyMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_EmptyAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "original text"  # kept


def test_translate_with_llm_runs_review_pass_when_enabled(monkeypatch):
    """translate_with_llm calls LLM twice per batch when review=True, once when False."""
    import vietdub.translate as t
    call_count = [0]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _CountingMessages:
        def create(self, **kwargs):
            call_count[0] += 1
            # Return a single-row response (works for both translate and review)
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

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_CountingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")]

    # With review=True (default): expect 2 LLM calls (translate + review)
    call_count[0] = 0
    translate_with_llm(segments=segments, context_bundle={}, settings=settings, review=True)
    assert call_count[0] == 2, f"Expected 2 LLM calls (translate + review), got {call_count[0]}"

    # With review=False: expect 1 LLM call (translate only)
    call_count[0] = 0
    translate_with_llm(segments=segments, context_bundle={}, settings=settings, review=False)
    assert call_count[0] == 1, f"Expected 1 LLM call (translate only), got {call_count[0]}"
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_translate.py::test_review_batch_for_length_returns_refined_translation tests/test_translate.py::test_review_batch_for_length_preserves_unchanged_translations tests/test_translate.py::test_review_batch_for_length_handles_partial_review_response tests/test_translate.py::test_review_batch_for_length_skips_empty_refinement tests/test_translate.py::test_translate_with_llm_runs_review_pass_when_enabled -v`

Expected: 5 fail (helper doesn't exist → ImportError on `_review_batch_for_length`).

- [ ] **Step 3: Add `_review_batch_for_length` function**

In `src/vietdub/translate.py`, after `_update_glossary` and `_cap_glossary` (around line 300), add:

```python
def _review_batch_for_length(
    rows: list[TranslationRow],
    context_bundle: dict,
    settings,
) -> list[TranslationRow]:
    """Review translations for length compliance and return refined rows.

    Sends each row's text_cn, text_vi, and target_vi_chars to the LLM with
    instruction to refine translations that significantly exceed target.
    Returns a new list of rows with updated text_vi (matching by segment_id).
    """
    import anthropic

    review_payload = {
        "instructions": [
            "Review these Vietnamese translations for length compliance.",
            "For each segment, target_vi_chars is the budget based on segment duration "
            "(Vietnamese ≈ 14 chars/sec + 10% buffer).",
            "Refine any translation that significantly exceeds its target_vi_chars "
            "(make it shorter while preserving meaning and tone).",
            "For translations already within budget or under budget, return unchanged.",
            "Preserve segment_id and original meaning; only adjust text_vi for length.",
            "Return JSON only with a top-level \"translations\" array.",
        ],
        "segments": [
            {
                "segment_id": row.segment_id,
                "text_cn": row.text_cn,
                "text_vi": row.text_vi,
                "target_vi_chars": _length_target_vi_chars(row.start_ms, row.end_ms),
            }
            for row in rows
        ],
    }

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=LLM_MAX_TOKENS,
            system=(
                "You are a Vietnamese translation editor refining for length. "
                "Return JSON only with a top-level \"translations\" array."
            ),
            messages=[{"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
        )
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIStatusError:
        raise
    except anthropic.APIConnectionError:
        raise

    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax review returned empty response (no text content blocks)")
    content = "".join(text_parts)
    refined_rows = parse_translation_response(content)

    refined_map: dict[str, str] = {r.segment_id: r.text_vi for r in refined_rows}

    output = []
    for row in rows:
        new_text = refined_map.get(row.segment_id, "").strip()
        if new_text:
            output.append(row.model_copy(update={"text_vi": new_text}))
        else:
            output.append(row)
    return output
```

- [ ] **Step 4: Wire `review` parameter into `_translate_one_batch_with_retry`**

In `src/vietdub/translate.py`, find `_translate_one_batch_with_retry` and update its signature + body to accept and use the `review` parameter:

Replace:
```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return _translate_one_batch(segments, context_bundle, settings, glossary=glossary)
        except (...)
```

With:
```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    review: bool = True,
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors.

    If review=True, runs a second LLM call per batch to refine translations
    for length compliance.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            rows = _translate_one_batch(segments, context_bundle, settings, glossary=glossary)
            if review:
                rows = _review_batch_for_length(rows, context_bundle, settings)
            return rows
        except (...)
```

(Only the signature and the `return` statement change — the `except (...)` block stays as-is.)

- [ ] **Step 5: Wire `review` parameter into `translate_with_llm`**

In `src/vietdub/translate.py`, find `translate_with_llm` and update its signature + the call to `_translate_one_batch_with_retry`:

Replace:
```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
) -> list[TranslationRow]:
```

With:
```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
    review: bool = True,
) -> list[TranslationRow]:
```

And find the call site inside the for loop (currently calls `_translate_one_batch_with_retry(batch, context_bundle, settings, glossary=glossary)`):

Replace with:
```python
        rows = _translate_one_batch_with_retry(
            batch, context_bundle, settings, glossary=glossary, review=review
        )
```

- [ ] **Step 6: Update `pipeline.py` and `cli.py` to pass `review` through**

In `src/vietdub/pipeline.py`, find `run_review_pipeline` (around line 52) and add a `review: bool = True` parameter:

```python
def run_review_pipeline(
    video: Path,
    jobs_dir: Path,
    series: str | None,
    settings,
    review: bool = True,
) -> Job:
    # ... existing code ...
    translations = translate_with_llm(
        merged,
        context_bundle=context_bundle,
        settings=typed_settings,
        review=review,
    )
```

In `src/vietdub/cli.py`, update the `run` command to accept `--no-review`:

Replace:
```python
@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", help="Pipeline mode: review or auto."),
    series: str | None = typer.Option(None, "--series", help="Series memory name."),
) -> None:
```

With:
```python
@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", help="Pipeline mode: review or auto."),
    series: str | None = typer.Option(None, "--series", help="Series memory name."),
    no_review: bool = typer.Option(False, "--no-review", help="Skip length review pass (saves LLM cost)."),
) -> None:
```

Then find the call to `run_review_pipeline` inside `run`:

Replace:
```python
    job = run_review_pipeline(video=video, jobs_dir=Path(settings.jobs_dir), series=series, settings=settings)
```

With:
```python
    job = run_review_pipeline(
        video=video,
        jobs_dir=Path(settings.jobs_dir),
        series=series,
        settings=settings,
        review=not no_review,
    )
```

- [ ] **Step 7: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_review_batch_for_length_returns_refined_translations tests/test_translate.py::test_review_batch_for_length_preserves_unchanged_translations tests/test_translate.py::test_review_batch_for_length_handles_partial_review_response tests/test_translate.py::test_review_batch_for_length_skips_empty_refinement tests/test_translate.py::test_translate_with_llm_runs_review_pass_when_enabled -v`

Expected: all 5 pass.

- [ ] **Step 8: Run full test suite to check for regressions**

Run: `pytest -v`

Expected: 145/145 pass (was 140 + 5 new = 145). Note: existing test `test_translate_with_llm_passes_glossary_to_subsequent_batches` may need a code review to ensure its mocked `_CapturingAnthropic` still works correctly with the new review call (it should, since the mock returns the same response shape).

- [ ] **Step 9: Commit**

```bash
cd C:\code\viethoaphimv4
git add src/vietdub/translate.py src/vietdub/cli.py src/vietdub/pipeline.py tests/test_translate.py
git commit -m "feat(translate): add opt-in length review pass (Spec 4 of 4)

Add _review_batch_for_length that runs after _translate_one_batch and
refines translations significantly exceeding target_vi_chars. Uses a
second LLM call per batch — opt-in via review=True parameter (default
True; CLI exposes --no-review flag for cost-sensitive operators).

Reduces noisy [LEN WARNING] stderr output from Spec 2 by having the
LLM explicitly optimize length in a dedicated pass. Cost: 2x LLM
calls per batch when review=True.

Final spec in the 4-spec translation quality series:
  - Spec 1: Better prompt (examples, tone, length targets)
  - Spec 2: Length check (warns on ±20% deviation)
  - Spec 3: Glossary consistency (cross-batch names)
  - Spec 4: Multi-pass review (refines to length target)"
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Section 1: `_review_batch_for_length` function | Task 1 Step 3 |
| Section 2: Wire `review` into `_translate_one_batch_with_retry` | Task 1 Step 4 |
| Section 2: Wire `review` into `translate_with_llm` | Task 1 Step 5 |
| Section 3: CLI `--no-review` flag | Task 1 Step 6 |
| Section 3: `pipeline.py` accepts `review` param | Task 1 Step 6 |
| 5 new tests (refine / preserve / partial / empty / call count) | Task 1 Step 1 |
| No regression in existing 140 tests | Task 1 Step 8 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- All code blocks show full implementation (function bodies, parameter lists, type annotations).
- Exact commands with expected output throughout.
- Tests show full bodies with mock setup, assertions, and teardown.

**3. Type consistency:**

- `_review_batch_for_length(rows: list[TranslationRow], context_bundle: dict, settings) -> list[TranslationRow]` — signature consistent across all 4 tests + 1 call site.
- `review: bool = True` parameter on `_translate_one_batch_with_retry` and `translate_with_llm` — consistent.
- `no_review: bool = False` CLI parameter — consistent with `review=not no_review` mapping.

**4. TDD discipline:**

- Step 1: write 5 failing tests.
- Step 2: verify all 5 fail (RED).
- Steps 3-6: implement (helper + 2 function signatures + CLI/pipeline plumbing).
- Step 7: verify all 5 pass (GREEN).
- Step 8: full suite regression check.
- Step 9: commit only after green.
