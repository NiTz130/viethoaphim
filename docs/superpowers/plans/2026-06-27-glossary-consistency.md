# Translation Glossary Consistency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain a per-job glossary of Chinese name → Vietnamese translations across batches in `translate_with_llm`, so the LLM uses consistent name translations throughout a job.

**Architecture:** Single source-file change to `src/vietdub/translate.py`. Make `translate_with_llm` stateful — maintain a `glossary: dict[str, str]` across batches. Initialize from `Names.txt` via a new `_load_initial_glossary` helper. After each batch, extract new terms via `_update_glossary` (CJK regex + position alignment). Cap at 50 entries via `_cap_glossary` (FIFO eviction). Pass glossary to each batch's prompt via a new optional `glossary` parameter on `build_translation_prompt`, which injects it as `consistency_terms`. Wire the new parameter through `_translate_one_batch` and `_translate_one_batch_with_retry`. Add 5 tests covering extraction, capitalization filter, glossary persistence across batches, FIFO cap, and prompt injection.

**Tech Stack:** Python 3.11+, pytest, regex (`re` stdlib).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-glossary-consistency-design.md` (authoritative for behavior).
- `_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")` matches 2-4 CJK Unified Ideographs (Chinese names).
- `_MAX_GLOSSARY_ENTRIES = 50` (module-level constant).
- `_load_initial_glossary()` reads `Names.txt` from `REFERENCE_DATA_DIR` env var (default `"data"`). Format: whitespace-separated `cn_name vi_name` pairs, one per line. Skip empty lines and lines starting with `#`. Graceful fallback to empty dict if file missing.
- `_update_glossary(glossary, rows, batch)`: for each row, find Chinese names via regex, align positionally with capitalized Vietnamese words (≥ 2 chars, first char uppercase).
- `_cap_glossary(glossary, max_entries=50)`: drop oldest entries FIFO when len > max.
- `build_translation_prompt(segments, context_bundle, glossary=None)`: adds `consistency_terms: [{source, target}, ...]` field to payload.
- `_translate_one_batch(segments, context_bundle, settings, glossary=None)`: passes glossary to `build_translation_prompt`.
- `_translate_one_batch_with_retry(segments, context_bundle, settings, glossary=None)`: passes glossary to `_translate_one_batch`.
- `translate_with_llm`: initializes glossary, passes to each batch, calls `_update_glossary` + `_cap_glossary` after each batch.
- Per-job scope: glossary state is local to each `translate_with_llm` invocation (resets between jobs).
- Existing 135 tests must continue to pass — `build_translation_prompt` called without `glossary` arg defaults to `None` → empty `consistency_terms`.
- TDD discipline: write failing tests, verify fail, implement, verify pass.
- No changes to: `parse_translation_response`, `_strip_markdown_fences`, `_warn_oversized_translations`, `_EXAMPLES`, `_length_target_vi_chars`, `LLM_*` constants, `_KNOWN_MODEL_OUTPUT_CAPS`.

---

### Task 1: Add glossary state + helpers + wire through pipeline

**Files:**
- Modify: `src/vietdub/translate.py` (add `_CHINESE_NAME_RE`, `_MAX_GLOSSARY_ENTRIES`, `_load_initial_glossary`, `_update_glossary`, `_cap_glossary`; update `build_translation_prompt` signature; update `_translate_one_batch` signature; update `_translate_one_batch_with_retry` signature; update `translate_with_llm` to maintain state)
- Modify: `tests/test_translate.py` (add 5 tests at the end)

**Interfaces:**
- Consumes: existing `TimedSegment`, `TranslationRow`, `build_translation_prompt`, `_translate_one_batch`, `_translate_one_batch_with_retry`, `translate_with_llm`, `Settings`.
- Produces: `translate_with_llm` returns same `list[TranslationRow]` (interface unchanged), but internally maintains per-call glossary state. Each batch's prompt includes accumulated glossary as `consistency_terms`.

- [ ] **Step 1: Add 5 failing tests**

Add the following to `tests/test_translate.py` at the end of the file:

```python
def test_extract_terms_from_translations_finds_chinese_names():
    """Heuristic extracts Chinese names → Vietnamese capitalized words."""
    from vietdub.translate import _update_glossary
    from vietdub.models import TranslationRow

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="长孙无忌 is here", text_vi="Trưởng Tôn đang ở đây",
            context_note="", status="draft",
        )
    ]
    batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="长孙无忌 is here")]
    glossary: dict = {}
    _update_glossary(glossary, rows, batch)
    assert "长孙无忌" in glossary
    assert glossary["长孙无忌"] == "Trưởng"


def test_extract_terms_from_translations_skips_non_capitalized():
    """Lowercase Vietnamese words don't get added to glossary."""
    from vietdub.translate import _update_glossary
    from vietdub.models import TranslationRow

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好世界", text_vi="xin chào bạn",
            context_note="", status="draft",
        )
    ]
    batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好世界")]
    glossary: dict = {}
    _update_glossary(glossary, rows, batch)
    assert glossary == {}


def test_translate_with_llm_passes_glossary_to_subsequent_batches(monkeypatch):
    """Glossary accumulates across batches and is passed forward."""
    import sys
    import types

    captured_prompts: list = []

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _CapturingMessages:
        """Returns a different translation each call to simulate cross-batch evolution."""
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            self.call_count += 1
            # First call returns 1 row with Chinese name "Trưởng"
            # Second call returns 1 row with Chinese name "Trưởng" again (consistency check)
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": f"m-{self.call_count:04d}",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "长孙无忌",
                        "text_vi": "Trưởng Tôn",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _CapturingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _CapturingMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_CapturingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    segments = [
        TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="长孙无忌"),
        TimedSegment(id="m-0002", start_ms=0, end_ms=1000, text="长孙无忌"),
    ]
    # With LLM_BATCH_SIZE=50, both fit in one batch. To force 2 batches,
    # we patch LLM_BATCH_SIZE temporarily.
    import vietdub.translate as t
    original_batch_size = t.LLM_BATCH_SIZE
    t.LLM_BATCH_SIZE = 1
    try:
        translate_with_llm(segments=segments, context_bundle={}, settings=settings)
    finally:
        t.LLM_BATCH_SIZE = original_batch_size

    assert len(captured_prompts) == 2
    # First batch's prompt: empty consistency_terms (glossary starts empty if Names.txt absent)
    first_payload = json.loads(captured_prompts[0])
    second_payload = json.loads(captured_prompts[1])
    # Second batch's prompt should include the glossary entry from batch 1
    sources = [t["source"] for t in second_payload.get("consistency_terms", [])]
    assert "长孙无忌" in sources
    targets = [t["target"] for t in second_payload.get("consistency_terms", [])]
    assert "Trưởng Tôn" in targets


def test_cap_glossary_drops_oldest_entries():
    """When glossary exceeds 50 entries, oldest are dropped FIFO."""
    from vietdub.translate import _cap_glossary

    glossary = {f"name_{i}": f"translation_{i}" for i in range(60)}
    _cap_glossary(glossary, max_entries=50)
    assert len(glossary) == 50
    assert "name_0" not in glossary  # oldest dropped
    assert "name_59" in glossary    # newest kept


def test_build_translation_prompt_includes_consistency_terms():
    """build_translation_prompt includes glossary as consistency_terms."""
    glossary = {"长孙无忌": "Trưởng Tôn"}
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        glossary=glossary,
    )
    payload = json.loads(prompt_json)
    assert "consistency_terms" in payload
    assert {"source": "长孙无忌", "target": "Trưởng Tôn"} in payload["consistency_terms"]
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_translate.py::test_extract_terms_from_translations_finds_chinese_names tests/test_translate.py::test_extract_terms_from_translations_skips_non_capitalized tests/test_translate.py::test_translate_with_llm_passes_glossary_to_subsequent_batches tests/test_translate.py::test_cap_glossary_drops_oldest_entries tests/test_translate.py::test_build_translation_prompt_includes_consistency_terms -v`

Expected: all 5 fail (helpers don't exist → ImportError, or `build_translation_prompt` doesn't accept `glossary` arg).

- [ ] **Step 3: Add `_CHINESE_NAME_RE`, `_MAX_GLOSSARY_ENTRIES`, and glossary helpers**

In `src/vietdub/translate.py`, near the existing constants (around line 95), add:

```python
_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")
_MAX_GLOSSARY_ENTRIES = 50
```

Then, somewhere near the top of the module (above `_translate_one_batch` for visibility), add:

```python
def _load_initial_glossary() -> dict[str, str]:
    """Load initial name glossary from reference data (Names.txt).

    Format: whitespace-separated 'cn_name vi_name' pairs, one per line.
    Skips empty lines and lines starting with '#'. Returns empty dict
    if the file is missing or unreadable.
    """
    glossary: dict[str, str] = {}
    try:
        ref_dir = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        names_file = ref_dir / "Names.txt"
        if names_file.exists():
            for line in names_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    glossary[parts[0]] = parts[1]
    except Exception:
        pass
    return glossary


def _update_glossary(
    glossary: dict[str, str],
    rows: list[TranslationRow],
    batch: list[TimedSegment],
) -> None:
    """Extract name_cn → name_vi pairs from completed batch and merge into glossary.

    Heuristic: for each segment, find Chinese name runs (2-4 CJK chars) and
    align them positionally with capitalized Vietnamese words.
    """
    for row, seg in zip(rows, batch):
        cn_names = _CHINESE_NAME_RE.findall(seg.text)
        if not cn_names:
            continue
        vi_words = row.text_vi.split()
        for i, cn_name in enumerate(cn_names):
            if i >= len(vi_words):
                break
            vi_word = vi_words[i].strip(".,!?;:")
            if len(vi_word) >= 2 and vi_word[0].isupper():
                glossary[cn_name] = vi_word


def _cap_glossary(
    glossary: dict[str, str],
    max_entries: int = _MAX_GLOSSARY_ENTRIES,
) -> None:
    """Cap glossary to max_entries by dropping oldest entries (FIFO)."""
    while len(glossary) > max_entries:
        oldest_key = next(iter(glossary))
        del glossary[oldest_key]
```

- [ ] **Step 4: Update `build_translation_prompt` to accept and inject `glossary`**

In `src/vietdub/translate.py`, replace the existing `build_translation_prompt` signature and body (currently lines 89-102) with:

```python
def build_translation_prompt(
    segments: list[TimedSegment],
    context_bundle: dict,
    glossary: dict[str, str] | None = None,
) -> str:
    payload = {
        "instructions": [
            "Return JSON only with a top-level \"translations\" array.",
            "Each translation must include segment_id, start_ms, end_ms, speaker, text_cn, text_vi, context_note, and status.",
            "Translate Chinese cartoon dialogue into natural Vietnamese.",
            "Use a silly, meme-friendly tone when the source is comedic.",
            "Aim for the target_vi_chars characters shown per segment (Vietnamese ≈ 14 chars/sec + 10% buffer).",
            "Preserve names and pronouns using the supplied context.",
            "Use the consistency_terms list below to translate recurring Chinese names consistently across batches.",
            "OUTPUT FORMAT (CORRECT): {\"translations\": [{\"segment_id\": \"m-0001\", \"text_vi\": \"...\", ...}]}",
            "OUTPUT FORMAT (INCORRECT — do NOT do this):",
            "  [{\"segment_id\": \"m-0001\", ...}]  (bare array, missing top-level object)",
            "  Wrapped in markdown fences or extra braces",
        ],
        "examples": _EXAMPLES,
        "consistency_terms": [
            {"source": cn, "target": vi} for cn, vi in (glossary or {}).items()
        ],
        "context": context_bundle,
        "segments": [
            {**segment.model_dump(), "target_vi_chars": _length_target_vi_chars(segment.start_ms, segment.end_ms)}
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
```

The only change: added `glossary: dict[str, str] | None = None` parameter and the `consistency_terms` field in payload, plus one new instruction line.

- [ ] **Step 5: Wire `glossary` through `_translate_one_batch` and `_translate_one_batch_with_retry`**

In `src/vietdub/translate.py`, update the signatures of both functions:

For `_translate_one_batch_with_retry` (around line 254):

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
        except api_connection_error as exc:
            last_exc = exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
        except api_status_error as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in {429, 500, 502, 503, 504}:
                if isinstance(exc, authentication_error):
                    raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
                raise RuntimeError(f"MiniMax HTTP {status}: {exc.message}") from exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))

    raise RuntimeError(
        f"MiniMax batch failed after {LLM_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc
```

(Only the signature changes — internal logic passes `glossary=glossary` to `_translate_one_batch`.)

For `_translate_one_batch` (around line 209), update signature and the `build_translation_prompt` call:

```python
def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
) -> list[TranslationRow]:
    # ... existing code unchanged through client.messages.create call ...

    user_prompt = build_translation_prompt(segments, context_bundle, glossary=glossary)

    # ... rest unchanged ...
```

- [ ] **Step 6: Update `translate_with_llm` to maintain glossary state**

In `src/vietdub/translate.py`, replace the batching loop section of `translate_with_llm` (currently lines 187-191):

Before:
```python
    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
    return all_rows
```

After:
```python
    glossary: dict[str, str] = dict(_load_initial_glossary())

    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        rows = _translate_one_batch_with_retry(batch, context_bundle, settings, glossary=glossary)
        all_rows.extend(rows)
        _update_glossary(glossary, rows, batch)
        _cap_glossary(glossary)

    return all_rows
```

Also add `import os` at the top of `translate.py` if not already imported (needed for `_load_initial_glossary`).

- [ ] **Step 7: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_extract_terms_from_translations_finds_chinese_names tests/test_translate.py::test_extract_terms_from_translations_skips_non_capitalized tests/test_translate.py::test_translate_with_llm_passes_glossary_to_subsequent_batches tests/test_translate.py::test_cap_glossary_drops_oldest_entries tests/test_translate.py::test_build_translation_prompt_includes_consistency_terms -v`

Expected: all 5 pass.

- [ ] **Step 8: Run full test suite to verify no regressions**

Run: `pytest -v`

Expected: 140/140 pass (was 135 + 5 new = 140). Existing tests:
- `build_translation_prompt` callers pass `glossary=None` (or no kwarg) → empty `consistency_terms` array in payload
- `_translate_one_batch` callers pass `glossary=None` (or no kwarg) → no behavior change
- `translate_with_llm` initializes glossary via `_load_initial_glossary` which gracefully returns empty dict if `Names.txt` missing

- [ ] **Step 9: Commit**

```bash
cd C:\code\viethoaphimv4
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): maintain glossary across batches for name consistency

Add per-job glossary state in translate_with_llm that accumulates
Chinese name → Vietnamese translations across batches:
  - Initializes from Names.txt via _load_initial_glossary
  - Auto-extracts new terms via _update_glossary (CJK regex +
    position-based alignment with capitalized Vietnamese words)
  - Capped at 50 entries via _cap_glossary (FIFO eviction)
  - Injected as consistency_terms in each subsequent batch's prompt

Helps the LLM translate recurring Chinese names consistently throughout
a job. Resets between jobs (per-job scope)."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Section 1: Glossary state in `translate_with_llm` | Task 1 Step 6 |
| Section 2: `_load_initial_glossary` from Names.txt | Task 1 Step 3 |
| Section 3: `_update_glossary` CJK regex + position alignment | Task 1 Step 3 |
| Section 4: `_cap_glossary` FIFO eviction at 50 entries | Task 1 Step 3 |
| Section 5: `build_translation_prompt` accepts glossary, injects `consistency_terms` | Task 1 Step 4 |
| Section 6: Wire glossary through `_translate_one_batch` + `_translate_one_batch_with_retry` | Task 1 Step 5 |
| 5 tests (extract finds / extract skips / persist across batches / cap drops / prompt includes) | Task 1 Step 1 |
| No regression in existing 135 tests | Task 1 Step 8 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- All code blocks show full implementation (regex, helpers, updated signatures, prompt format).
- Exact commands with expected output throughout.
- Tests show full bodies, not "similar to".

**3. Type consistency:**

- `_CHINESE_NAME_RE = re.compile(...)` — module-level, type `re.Pattern[str]`.
- `_MAX_GLOSSARY_ENTRIES = 50` — module-level, type `int`.
- `_load_initial_glossary() -> dict[str, str]` — signature consistent across all usages.
- `_update_glossary(glossary: dict[str, str], rows: list[TranslationRow], batch: list[TimedSegment]) -> None` — signature consistent across test + call site.
- `_cap_glossary(glossary: dict[str, str], max_entries: int = _MAX_GLOSSARY_ENTRIES) -> None` — signature consistent.
- `build_translation_prompt(segments, context_bundle, glossary: dict[str, str] | None = None) -> str` — new optional kwarg, default None for backward compat.
- `_translate_one_batch(..., glossary: dict[str, str] | None = None) -> list[TranslationRow]` — new optional kwarg.
- `_translate_one_batch_with_retry(..., glossary: dict[str, str] | None = None) -> list[TranslationRow]` — new optional kwarg.

**4. TDD discipline:**

- Step 1: write 5 failing tests.
- Step 2: verify all 5 fail (RED).
- Steps 3-6: implement (constants, helpers, prompt update, wire through).
- Step 7: verify all 5 pass (GREEN).
- Step 8: full suite regression check.
- Step 9: commit only after green.
