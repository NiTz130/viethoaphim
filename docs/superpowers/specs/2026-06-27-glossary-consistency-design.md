# Translation Glossary Consistency — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (state across batches in `translate_with_llm`) + `tests/test_translate.py`
**Status:** Approved (design), pending implementation

## Context

The viethoaphimv4 translation pipeline splits segments into batches of `LLM_BATCH_SIZE = 50`. Each batch is sent to the LLM independently. Without shared state, the LLM may translate the same Chinese name inconsistently across batches:
- Batch 1: 长孙无忌 → "Trưởng Tôn Vô Kỵ"
- Batch 2: 长孙无忌 → "Trưởng Tôn Bất Kỵ" (different!)
- Batch 3: 长孙无忌 → "Trưởng Tôn Vô Kỵ" (back to first)

The existing context mechanism (`build_context_bundle` in `src/vietdub/context.py`) reads `Names.txt` from reference data and passes it to every batch. But `Names.txt` is static — it doesn't include names invented by the LLM during the current job (e.g., minor character names not in the reference file).

This is Spec 3 of 4 (Prompt → Length → Glossary → Multi-pass Review).

## Goal

Maintain a per-job glossary that:
1. Initializes from existing reference data (`Names.txt` via `build_reference_context`).
2. Grows as the LLM invents or settles on new name translations during a job.
3. Is passed to each subsequent batch in the prompt as `consistency_terms`, so the LLM uses the same translation for recurring Chinese names.

## Non-goals

- Cross-job glossary persistence — glossary resets between jobs (per user scope choice).
- Detecting ALL terminology, not just names — focus on Chinese proper nouns for now.
- Auto-correcting post-LLM inconsistencies (use spec 4 multi-pass review for that).
- Replacing the existing reference data — `Names.txt` is still consulted, just augmented.

## Design

### Section 1: Glossary state in `translate_with_llm`

`translate_with_llm` (currently lines 156-191) becomes stateful — maintains a `glossary: dict[str, str]` across batches:

```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
) -> list[TranslationRow]:
    """Translate segments by batching them into LLM calls.

    Maintains a glossary across batches for name consistency. Initializes
    from Names.txt (reference data), grows as the LLM invents new name
    translations during the job. Glossary resets between jobs.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for translation")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is required for translation")

    cap = KNOWN_MODEL_OUTPUT_CAPS.get(settings.llm_model)
    if cap is not None and LLM_MAX_TOKENS > cap:
        raise RuntimeError(...)
    if cap is None:
        print(...)

    glossary: dict[str, str] = dict(_load_initial_glossary())

    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        rows = _translate_one_batch_with_retry(batch, context_bundle, settings, glossary=glossary)
        all_rows.extend(rows)
        _update_glossary(glossary, rows, batch)
        if len(glossary) > 50:
            _cap_glossary(glossary, max_entries=50)

    return all_rows
```

`_translate_one_batch_with_retry` gets a new `glossary` parameter passed through to `build_translation_prompt`.

### Section 2: `_load_initial_glossary` — Names.txt extraction

Pull name pairs from `Names.txt` (already loaded by `build_reference_context` via `derive_characters`):

```python
def _load_initial_glossary() -> dict[str, str]:
    """Load initial name glossary from reference data (Names.txt)."""
    glossary: dict[str, str] = {}
    try:
        ref_dir = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        names_file = ref_dir / "Names.txt"
        if names_file.exists():
            for line in names_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format varies; assume tab/comma/space-separated pairs
                parts = line.split(None, 1)  # split on first whitespace
                if len(parts) == 2:
                    glossary[parts[0]] = parts[1]
    except Exception:
        pass  # Reference data is optional
    return glossary
```

(Falls back gracefully if `Names.txt` is missing — auto-extracted terms still work.)

### Section 3: `_update_glossary` — extract from completed batch

Use regex + position-based alignment to find Chinese names and their Vietnamese translations:

```python
_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")


def _update_glossary(glossary: dict[str, str], rows: list[TranslationRow], batch: list[TimedSegment]) -> None:
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
            # Capitalized Vietnamese = proper noun
            if len(vi_word) >= 2 and vi_word[0].isupper():
                glossary[cn_name] = vi_word
```

### Section 4: `_cap_glossary` — FIFO eviction

Cap at 50 entries to prevent prompt overflow. Drop oldest entries first (FIFO):

```python
_MAX_GLOSSARY_ENTRIES = 50


def _cap_glossary(glossary: dict[str, str], max_entries: int = _MAX_GLOSSARY_ENTRIES) -> None:
    """Cap glossary to max_entries by dropping oldest entries (FIFO)."""
    while len(glossary) > max_entries:
        oldest_key = next(iter(glossary))
        del glossary[oldest_key]
```

### Section 5: `build_translation_prompt` accepts glossary

Add `glossary` parameter (optional, defaults to None) and inject as `consistency_terms`:

```python
def build_translation_prompt(
    segments: list[TimedSegment],
    context_bundle: dict,
    glossary: dict[str, str] | None = None,
) -> str:
    payload = {
        "instructions": [
            # ... existing instructions ...
            "Use the consistency_terms list below to translate recurring Chinese names consistently across batches.",
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

### Section 6: Wire glossary through `_translate_one_batch_with_retry` and `_translate_one_batch`

Add `glossary` parameter to both functions:

```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
) -> list[TranslationRow]:
    # ... existing logic ...
    return _translate_one_batch(segments, context_bundle, settings, glossary=glossary)


def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
) -> list[TranslationRow]:
    # ... existing logic ...
    user_prompt = build_translation_prompt(segments, context_bundle, glossary=glossary)
    # ... rest unchanged ...
```

## Test plan

Add 5 tests to `tests/test_translate.py`:

1. **`test_extract_terms_from_translations_finds_chinese_names`**
   ```python
   def test_extract_terms_from_translations_finds_chinese_names():
       """Heuristic extracts Chinese names → Vietnamese capitalized words."""
       rows = [
           TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000,
                           speaker=None, text_cn="长孙无忌 is here",
                           text_vi="Trưởng Tôn đang ở đây",
                           context_note="", status="draft")
       ]
       batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="长孙无忌 is here")]
       glossary = {}
       _update_glossary(glossary, rows, batch)
       assert "长孙无忌" in glossary
       assert glossary["长孙无忌"] == "Trưởng"
   ```

2. **`test_extract_terms_from_translations_skips_non_capitalized`**
   ```python
   def test_extract_terms_from_translations_skips_non_capitalized():
       """Lowercase Vietnamese words don't get added to glossary."""
       rows = [
           TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000,
                           speaker=None, text_cn="你好世界",
                           text_vi="xin chào bạn",
                           context_note="", status="draft")
       ]
       batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好世界")]
       glossary = {}
       _update_glossary(glossary, rows, batch)
       assert glossary == {}
   ```

3. **`test_translate_with_llm_passes_glossary_to_subsequent_batches`**
   ```python
   def test_translate_with_llm_passes_glossary_to_subsequent_batches(monkeypatch):
       """Glossary accumulates across batches and is passed forward."""
       # Mock anthropic; capture user_prompt from second call
       captured_prompts = []
       class _CapturingMessages:
           def create(self, **kwargs):
               captured_prompts.append(kwargs["messages"][0]["content"])
               # Return one translation per batch
               # ... (similar to existing _FakeAnthropic pattern)
       # Run with 51 segments → 2 batches; assert batch 2's prompt contains terms from batch 1
   ```

4. **`test_cap_glossary_drops_oldest_entries`**
   ```python
   def test_cap_glossary_drops_oldest_entries():
       """When glossary exceeds 50 entries, oldest are dropped FIFO."""
       glossary = {f"name_{i}": f"translation_{i}" for i in range(60)}
       _cap_glossary(glossary, max_entries=50)
       assert len(glossary) == 50
       assert "name_0" not in glossary  # oldest dropped
       assert "name_59" in glossary    # newest kept
   ```

5. **`test_build_translation_prompt_includes_consistency_terms`**
   ```python
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

## Risk

**Medium.** Heuristic extraction (regex + position alignment) is imperfect. Could mis-extract (e.g., "xem" mapped to "Xem" because position 0 is Chinese). Mitigation: validation requires capitalized Vietnamese (proper noun heuristic) + length ≥ 2 chars. False positives are tolerable — at worst, glossary contains a wrong entry that the LLM may then propagate. False negatives (missed real names) are worse — but the LLM still has the original Names.txt as fallback.

## Rollback

Single source file change + tests. Revert the implementation commit if operators report regressions.

## Implementation notes

- `_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")` matches 2-4 CJK Unified Ideographs in a row — typical for Chinese names.
- The `2-4` length window matches 2-3 character Chinese names (most common) and 4-character names (e.g., 长孙无忌 is 4 chars).
- Glossary state is per-`translate_with_llm` call (per-job), not module-level. This means parallel jobs don't share state.
- `_load_initial_glossary` is robust to missing Names.txt — falls back to empty dict, and the auto-extract still works.
- `_MAX_GLOSSARY_ENTRIES = 50` is a module-level constant, tunable.
- Existing tests for `build_translation_prompt` continue to pass when called without `glossary` argument (defaults to None → empty `consistency_terms` array).
- No changes to: `parse_translation_response`, `_strip_markdown_fences`, `_warn_oversized_translations`, `_EXAMPLES`, `_length_target_vi_chars`, `LLM_*` constants.
