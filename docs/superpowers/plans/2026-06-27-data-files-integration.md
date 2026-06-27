# Data Files Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load 3 high-utility reference data files (Pronouns, LuatNhan, IgnoredChinesePhrases) into the translation prompt as inline sections, improving tone consistency, idiom fidelity, and output cleanliness.

**Architecture:** Single source-file change to `src/vietdub/translate.py`. Add 3 private helper functions (`_load_pronouns`, `_load_phrase_patterns`, `_load_ignore_list`) that read from `data/` via `find_reference_data_dir` (Spec 3 helper). Update `build_translation_prompt` to accept and inject 3 new payload fields (`pronoun_guide`, `phrase_patterns`, `ignore_list`) plus 3 new instructions. Thread the 3 new sections through `_translate_one_batch`, `_translate_one_batch_with_retry`, and `translate_with_llm`. Add 4 tests in `tests/test_translate.py`.

**Tech Stack:** Python 3.11+, pytest, existing `find_reference_data_dir` + `load_dictionary_entries` from `src/vietdub/reference.py`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-data-files-integration-design.md` (authoritative).
- 3 new helper functions: `_load_pronouns(ref_dir) -> list[dict[str, str]]`, `_load_phrase_patterns(ref_dir) -> list[dict[str, str]]`, `_load_ignore_list(ref_dir) -> list[str]`.
- `_load_pronouns` reuses `load_dictionary_entries` (handles `=` separator).
- `_load_phrase_patterns` parses `cn{0}=vi{0}` format manually (custom parser).
- `_load_ignore_list` returns list of raw strings (one per non-blank line).
- All 3 helpers return empty list when file missing (graceful fallback).
- 3 new optional parameters on `build_translation_prompt`: `pronoun_guide`, `phrase_patterns`, `ignore_list` (default None → empty list in payload).
- 3 new instructions appended to existing `instructions` array in `build_translation_prompt`.
- 3 new payload fields: `pronoun_guide`, `phrase_patterns`, `ignore_list` (each = list, default empty).
- 3 new optional parameters threaded through `_translate_one_batch`, `_translate_one_batch_with_retry`, `translate_with_llm`.
- `translate_with_llm` loads 3 data sources ONCE (not per batch) — efficient since data is static per job.
- Existing 145 tests must continue to pass — new payload fields default to empty (backward compat).
- TDD discipline: write failing tests, verify fail, implement, verify pass.
- No changes to: `parse_translation_response`, `_strip_markdown_fences`, `_warn_oversized_translations`, `_EXAMPLES`, `_length_target_vi_chars`, retry logic, `_review_batch_for_length`, batching, CLI, `_load_initial_glossary` (Names.txt loader from Spec 3).

---

### Task 1: Add 3 helpers + wire through pipeline + 4 tests

**Files:**
- Modify: `src/vietdub/translate.py` (add 3 helper functions near `_load_initial_glossary`; update 4 function signatures; add 3 instructions; add 3 payload fields)
- Modify: `tests/test_translate.py` (add 4 tests at the end)

**Interfaces:**
- Consumes: existing `find_reference_data_dir`, `load_dictionary_entries`, `Path`, `os`, existing `build_translation_prompt`, `_translate_one_batch`, `_translate_one_batch_with_retry`, `translate_with_llm`.
- Produces: 3 new prompt payload fields populated from reference data files. `translate_with_llm` signature gains 3 new optional kwargs (all default None → no behavior change for callers).

- [ ] **Step 1: Add 4 failing tests**

Add the following to `tests/test_translate.py` at the end of the file:

```python
def test_build_translation_prompt_includes_pronoun_guide():
    """build_translation_prompt payload includes pronoun_guide section."""
    pronoun_guide = [{"source": "你自己", "target": "chính ngươi"}]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        pronoun_guide=pronoun_guide,
    )
    payload = json.loads(prompt_json)
    assert "pronoun_guide" in payload
    assert payload["pronoun_guide"] == pronoun_guide


def test_build_translation_prompt_includes_phrase_patterns():
    """build_translation_prompt payload includes phrase_patterns section with placeholders."""
    phrase_patterns = [{"source": "与{0}为敌为友", "target": "cùng {0} là địch là bạn"}]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        phrase_patterns=phrase_patterns,
    )
    payload = json.loads(prompt_json)
    assert "phrase_patterns" in payload
    assert payload["phrase_patterns"] == phrase_patterns
    # Placeholder preserved
    assert "{0}" in payload["phrase_patterns"][0]["source"]


def test_build_translation_prompt_includes_ignore_list():
    """build_translation_prompt payload includes ignore_list section."""
    ignore_list = ["( 小说 《 九 鼎记 ... )", "(未 完 待续)..."]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        ignore_list=ignore_list,
    )
    payload = json.loads(prompt_json)
    assert "ignore_list" in payload
    assert payload["ignore_list"] == ignore_list


def test_load_helpers_resilient_to_missing_files(tmp_path):
    """All 3 loaders return empty list when file missing (no exception)."""
    from vietdub.translate import _load_pronouns, _load_phrase_patterns, _load_ignore_list

    # tmp_path doesn't have any of the data files
    assert _load_pronouns(tmp_path) == []
    assert _load_phrase_patterns(tmp_path) == []
    assert _load_ignore_list(tmp_path) == []
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_translate.py::test_build_translation_prompt_includes_pronoun_guide tests/test_translate.py::test_build_translation_prompt_includes_phrase_patterns tests/test_translate.py::test_build_translation_prompt_includes_ignore_list tests/test_translate.py::test_load_helpers_resilient_to_missing_files -v`

Expected: 4 fail (helpers don't exist → ImportError; `build_translation_prompt` doesn't accept new kwargs → TypeError).

- [ ] **Step 3: Add 3 helper functions**

In `src/vietdub/translate.py`, after `_load_initial_glossary` (around line 280, after the `except Exception: glossary = {}` block), add:

```python
def _load_pronouns(ref_dir: Path) -> list[dict[str, str]]:
    """Load pronoun guide from Pronouns.txt. Format: cn=vi per line.

    Uses the shared load_dictionary_entries helper (handles = separator).
    Returns empty list if file missing.
    """
    return load_dictionary_entries(ref_dir / "Pronouns.txt")


def _load_phrase_patterns(ref_dir: Path) -> list[dict[str, str]]:
    """Load phrase patterns from LuatNhan.txt. Format: cn{0}=vi{0} per line.

    Returns list of {source, target} dicts. Empty list if file missing.
    """
    patterns: list[dict[str, str]] = []
    path = ref_dir / "LuatNhan.txt"
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        patterns.append({"source": source.strip(), "target": target.strip()})
    return patterns


def _load_ignore_list(ref_dir: Path) -> list[str]:
    """Load Chinese boilerplate phrases to ignore from IgnoredChinesePhrases.txt.

    Each line is a long Chinese phrase (boilerplate from web novel scraping).
    Returns list of raw phrases. Empty list if file missing.
    """
    ignore_path = ref_dir / "IgnoredChinesePhrases.txt"
    if not ignore_path.exists():
        return []
    return [line.strip() for line in ignore_path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: Update `build_translation_prompt` to accept and inject the 3 new sections**

In `src/vietdub/translate.py`, find `build_translation_prompt` (around line 160) and replace its signature + body:

Before:
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

After:
```python
def build_translation_prompt(
    segments: list[TimedSegment],
    context_bundle: dict,
    glossary: dict[str, str] | None = None,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
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
            "Use the pronoun_guide to choose appropriate Vietnamese pronouns based on context (modern vs. historical, formal vs. casual).",
            "Use the phrase_patterns as templates — replace {0} with the appropriate referent when translating matching Chinese phrases.",
            "If the source text contains any phrase from the ignore_list, exclude it from the translation (it's boilerplate from web scraping, not actual content).",
            "OUTPUT FORMAT (CORRECT): {\"translations\": [{\"segment_id\": \"m-0001\", \"text_vi\": \"...\", ...}]}",
            "OUTPUT FORMAT (INCORRECT — do NOT do this):",
            "  [{\"segment_id\": \"m-0001\", ...}]  (bare array, missing top-level object)",
            "  Wrapped in markdown fences or extra braces",
        ],
        "examples": _EXAMPLES,
        "consistency_terms": [
            {"source": cn, "target": vi} for cn, vi in (glossary or {}).items()
        ],
        "pronoun_guide": pronoun_guide or [],
        "phrase_patterns": phrase_patterns or [],
        "ignore_list": ignore_list or [],
        "context": context_bundle,
        "segments": [
            {**segment.model_dump(), "target_vi_chars": _length_target_vi_chars(segment.start_ms, segment.end_ms)}
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
```

- [ ] **Step 5: Wire the 3 new sections through `_translate_one_batch`**

In `src/vietdub/translate.py`, find `_translate_one_batch` (around line 209). Update its signature and the `build_translation_prompt` call.

Before signature:
```python
def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
) -> list[TranslationRow]:
```

After signature:
```python
def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
) -> list[TranslationRow]:
```

Find the `user_prompt = build_translation_prompt(segments, context_bundle, glossary=glossary)` line inside `_translate_one_batch` and update:

```python
    user_prompt = build_translation_prompt(
        segments, context_bundle, glossary=glossary,
        pronoun_guide=pronoun_guide,
        phrase_patterns=phrase_patterns,
        ignore_list=ignore_list,
    )
```

- [ ] **Step 6: Wire the 3 new sections through `_translate_one_batch_with_retry`**

In `src/vietdub/translate.py`, find `_translate_one_batch_with_retry` (around line 453). Update its signature and the call to `_translate_one_batch`.

Before signature:
```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    review: bool = True,
) -> list[TranslationRow]:
```

After signature:
```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    review: bool = True,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
) -> list[TranslationRow]:
```

Find the call to `_translate_one_batch` inside the try block:

```python
            rows = _translate_one_batch(
                segments, context_bundle, settings, glossary=glossary,
                pronoun_guide=pronoun_guide,
                phrase_patterns=phrase_patterns,
                ignore_list=ignore_list,
            )
```

(Only update if review=True, before the if review block; same indentation level.)

- [ ] **Step 7: Wire the 3 new sections through `translate_with_llm`**

In `src/vietdub/translate.py`, find `translate_with_llm` (around line 203). Add 3 new parameters to its signature and load the 3 data sources before the batching loop.

Update signature:
```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
    review: bool = True,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
) -> list[TranslationRow]:
```

After the `glossary: dict[str, str] = dict(_load_initial_glossary())` line and `_cap_glossary(glossary)` line, add the 3 loads (with try/except fallback):

```python
    # Load 3 reference data sections (loaded once, reused across batches)
    try:
        ref_root = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        ref_dir = find_reference_data_dir(ref_root)
        pronoun_guide = _load_pronouns(ref_dir)
        phrase_patterns = _load_phrase_patterns(ref_dir)
        ignore_list = _load_ignore_list(ref_dir)
    except Exception:
        pronoun_guide = []
        phrase_patterns = []
        ignore_list = []
```

Find the call to `_translate_one_batch_with_retry` inside the for loop and update:

```python
        rows = _translate_one_batch_with_retry(
            batch, context_bundle, settings,
            glossary=glossary, review=review,
            pronoun_guide=pronoun_guide,
            phrase_patterns=phrase_patterns,
            ignore_list=ignore_list,
        )
```

- [ ] **Step 8: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_build_translation_prompt_includes_pronoun_guide tests/test_translate.py::test_build_translation_prompt_includes_phrase_patterns tests/test_translate.py::test_build_translation_prompt_includes_ignore_list tests/test_translate.py::test_load_helpers_resilient_to_missing_files -v`

Expected: all 4 pass.

- [ ] **Step 9: Run full test suite to verify no regressions**

Run: `pytest -q`

Expected: 149/149 pass (was 145 + 4 new = 149). Existing tests still pass — `build_translation_prompt` called without the 3 new kwargs defaults to None → empty lists in payload (backward compat).

- [ ] **Step 10: Commit**

```bash
cd C:\code\viethoaphimv4
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): integrate pronoun/phrase/ignore reference data

Add 3 helper functions (_load_pronouns, _load_phrase_patterns,
_load_ignore_list) that load from data/ via find_reference_data_dir
(Spec 3 helper). Update build_translation_prompt to accept 3 new
optional kwargs and inject as pronoun_guide, phrase_patterns,
ignore_list sections in user prompt payload. Thread through
_translate_one_batch, _translate_one_batch_with_retry, and
translate_with_llm. Add 4 tests covering inclusion + graceful fallback.

Sources:
  - Pronouns.txt (28) — tone/style guide for xưng hô
  - LuatNhan.txt (225) — phrase patterns with {0} placeholders
  - IgnoredChinesePhrases.txt (279) — boilerplate to skip

Excluded: LacViet, ChinesePhienAmWords (marginal), VietPhrase.txt
(data contamination), VietPhrase (2).txt (out of scope)."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Section 1: 3 helper functions | Task 1 Step 3 |
| Section 2: Update `build_translation_prompt` (3 params + 3 fields + 3 instructions) | Task 1 Step 4 |
| Section 3: Wire in `translate_with_llm` | Task 1 Step 7 |
| Section 4: Thread through `_translate_one_batch_with_retry` + `_translate_one_batch` | Task 1 Steps 5-6 |
| 4 tests (pronoun / phrase / ignore / missing file) | Task 1 Step 1 |
| No regression in existing 145 tests | Task 1 Step 9 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- All code blocks show full implementation (helper bodies, function signatures, payload structure).
- Exact commands with expected output throughout.
- Tests show full bodies with setup, assertions.

**3. Type consistency:**

- `_load_pronouns(ref_dir: Path) -> list[dict[str, str]]` — consistent across test + 1 call site (`translate_with_llm`).
- `_load_phrase_patterns(ref_dir: Path) -> list[dict[str, str]]` — consistent.
- `_load_ignore_list(ref_dir: Path) -> list[str]` — consistent.
- `pronoun_guide: list[dict[str, str]] | None = None` parameter — consistent across 4 function signatures (build_translation_prompt, _translate_one_batch, _translate_one_batch_with_retry, translate_with_llm).
- Same for `phrase_patterns` and `ignore_list`.

**4. TDD discipline:**

- Step 1: write 4 failing tests.
- Step 2: verify all 4 fail (RED).
- Steps 3-7: implement (helpers + 4 function signatures updated + load logic).
- Step 8: verify all 4 pass (GREEN).
- Step 9: full suite regression check.
- Step 10: commit only after green.
