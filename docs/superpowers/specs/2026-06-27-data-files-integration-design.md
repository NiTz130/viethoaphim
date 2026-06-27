# Data Files Integration (Pronouns, Phrase Patterns, Ignore List) — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (3 new helper functions + `build_translation_prompt` updates) + `tests/test_translate.py`
**Status:** Approved (design), pending implementation

## Context

The viethoaphimv4 translation pipeline currently only loads `Names.txt` from the reference data directory (`data/Data của thtgiang (đọc README)/`). The directory contains 8 additional text files that are currently ignored:

| File | Lines | Content |
|---|---|---|
| `Names.txt` | 159,162 | Person/place name translations (already loaded as glossary — Spec 3) |
| `Pronouns.txt` | 28 | Chinese → Vietnamese pronouns (e.g., `你自己` → `chính ngươi`) |
| `LuatNhan.txt` | 225 | Phrase patterns with `{0}` placeholders (e.g., `与{0}为敌为友` → `cùng {0} là địch là bạn`) |
| `IgnoredChinesePhrases.txt` | 279 | Boilerplate Chinese text from web novels to skip in OCR/translation |
| `ChinesePhienAmEnglishWords.txt` | 313 | Marginal — LLM already knows English |
| `ChinesePhienAmWords.txt` | 12,630 | Marginal — LLM already knows HanViet |
| `LacViet.txt` | 66,449 | Marginal — LLM already knows HanViet readings |
| `VietPhrase.txt` | 870,114 | Data contamination (English/German phrases) — out of scope |
| `VietPhrase (2).txt` | 274,774 | Useful but very large — out of scope (future spec) |

This spec adds the 3 high-utility files (Pronouns, LuatNhan, IgnoredChinesePhrases) as inline sections in the user prompt.

This is Spec 5 — extends the 4-spec "improve translation quality" series with broader reference data integration.

## Goal

Add inline `pronoun_guide`, `phrase_patterns`, and `ignore_list` sections to the user prompt payload, each populated from a corresponding reference data file. The LLM receives these as explicit guidance, improving:
- **Tone consistency** (pronouns: `ngươi` vs `bạn` based on context)
- **Idiom fidelity** (phrase patterns with placeholders)
- **Output cleanliness** (skip boilerplate OCR noise)

## Non-goals

- Loading `LacViet.txt` (66k) or `ChinesePhienAmWords.txt` (12k) — LLM already knows HanViet, marginal value
- Loading `VietPhrase.txt` (870k) — data contamination
- Loading `VietPhrase (2).txt` (274k) — out of scope (future spec for phrase database)
- Changing `Names.txt` glossary loading behavior from Spec 3
- Source code changes to OCR step (IgnoredChinesePhrases is for the LLM to skip, not for OCR filtering)
- New CLI flags for these data sources

## Design

### Section 1: 3 new helper functions

Add 3 private helpers in `src/vietdub/translate.py` (near `_load_initial_glossary`):

```python
def _load_pronouns(ref_dir: Path) -> list[dict[str, str]]:
    """Load pronoun guide from Pronouns.txt. Format: cn=vi per line.
    Uses the shared load_dictionary_entries helper (handles = separator).
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

### Section 2: Update `build_translation_prompt` to accept and inject the 3 new sections

Add 3 new optional parameters to `build_translation_prompt`:

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
            # ... existing 8 instructions ...
            # Add 3 new:
            "Use the pronoun_guide to choose appropriate Vietnamese pronouns based on context (modern vs. historical, formal vs. casual).",
            "Use the phrase_patterns as templates — replace {0} with the appropriate referent when translating matching Chinese phrases.",
            "If the source text contains any phrase from the ignore_list, exclude it from the translation (it's boilerplate from web scraping, not actual content).",
        ],
        "examples": _EXAMPLES,
        "consistency_terms": [
            {"source": cn, "target": vi} for cn, vi in (glossary or {}).items()
        ],
        "pronoun_guide": pronoun_guide or [],
        "phrase_patterns": phrase_patterns or [],
        "ignore_list": ignore_list or [],
        "context": context_bundle,
        "segments": [...],
    }
    return json.dumps(payload, ensure_ascii=False)
```

The 3 new parameters default to `None` → empty lists in payload (backward compat with existing callers).

### Section 3: Wire the 3 helpers in `translate_with_llm`

In `translate_with_llm` (currently lines 203-244), load the 3 data sources once and pass to each batch's prompt:

```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
    review: bool = True,
) -> list[TranslationRow]:
    # ... existing validation ...

    glossary: dict[str, str] = dict(_load_initial_glossary())
    _cap_glossary(glossary)

    # NEW: load 3 reference data sections (loaded once, reused across batches)
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

    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        rows = _translate_one_batch_with_retry(
            batch, context_bundle, settings,
            glossary=glossary, review=review,
            pronoun_guide=pronoun_guide,
            phrase_patterns=phrase_patterns,
            ignore_list=ignore_list,
        )
        all_rows.extend(rows)
        _update_glossary(glossary, rows, batch)
        _cap_glossary(glossary)
    return all_rows
```

### Section 4: Thread the 3 new sections through `_translate_one_batch_with_retry` and `_translate_one_batch`

Update both function signatures to accept and forward the 3 new sections:

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
    # ... pass pronoun_guide, phrase_patterns, ignore_list to _translate_one_batch ...


def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
) -> list[TranslationRow]:
    # ... pass pronoun_guide, phrase_patterns, ignore_list to build_translation_prompt ...
    user_prompt = build_translation_prompt(
        segments, context_bundle, glossary=glossary,
        pronoun_guide=pronoun_guide, phrase_patterns=phrase_patterns, ignore_list=ignore_list,
    )
```

## Test plan

Add 4 tests to `tests/test_translate.py`:

1. **`test_build_translation_prompt_includes_pronoun_guide`**
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
   ```

2. **`test_build_translation_prompt_includes_phrase_patterns`**
   ```python
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
   ```

3. **`test_build_translation_prompt_includes_ignore_list`**
   ```python
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
   ```

4. **`test_load_helpers_resilient_to_missing_files`**
   ```python
   def test_load_helpers_resilient_to_missing_files(monkeypatch, tmp_path):
       """All 3 loaders return empty list when file missing (no exception)."""
       from vietdub.translate import _load_pronouns, _load_phrase_patterns, _load_ignore_list
       
       # tmp_path doesn't have any of the data files
       assert _load_pronouns(tmp_path) == []
       assert _load_phrase_patterns(tmp_path) == []
       assert _load_ignore_list(tmp_path) == []
   ```

## Risk

**Low.** Additive change — only adds new payload fields and helper functions. No modifications to existing logic. Existing 145 tests should continue to pass since the new payload fields default to empty lists.

## Rollback

Single source file change + tests. Revert the implementation commit if operators report regressions.

## Implementation notes

- The 3 helpers use `find_reference_data_dir` (Spec 3) to locate the data files, consistent with `_load_initial_glossary`.
- The 3 new sections are loaded ONCE in `translate_with_llm` (not per-batch) — efficient since data is static per job.
- All 3 helpers gracefully return empty lists on missing files / parse errors — robustness over strict validation.
- The 3 new instructions are added to the existing `instructions` array (preserves order).
- No new dependencies or imports.
- No changes to `parse_translation_response`, `_strip_markdown_fences`, `_warn_oversized_translations`, `_EXAMPLES`, `_length_target_vi_chars`, retry logic, `_review_batch_for_length`, batching, or CLI.
- Total prompt size increase: ~5-10k characters (28 + 225 + 279 lines × ~30 chars each = ~16k chars, but many entries are short, so actual increase is smaller). Still well under `max_tokens=8192` output cap.
- The 3 new sections do NOT participate in glossary capping (unlike `_update_glossary` and `_cap_glossary` which manage the 50-entry cap). They're loaded fresh per job and stay stable.
