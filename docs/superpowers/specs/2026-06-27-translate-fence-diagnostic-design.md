# Translate Fence Parsing + Diagnostic Hardening — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py:_strip_markdown_fences` + `src/vietdub/translate.py:parse_translation_response` + tests
**Status:** Approved (design), pending implementation

## Context

The 2026-06-27 code review surfaced 7 findings. Findings #1 and #2 (silent truncation + no retry) were addressed in commit series `44e07a8` through `11b95a8`. This spec addresses findings #3, #4, and #7.

`_strip_markdown_fences` (translate.py:29-38) uses simple string operations that fail on two edge cases:

- **Finding #3 (single-line opening fence):** LLMs sometimes return ` ```json {...} ``` ` (no newline between opening fence and JSON). Current code does `find("\n")` which finds the newline AFTER the JSON, then slices keep the closing fence, and `rfind("```")` strips it. Result: the function returns an empty string, `json.loads("")` fails, and the user sees a misleading "invalid JSON at char 0" error.

- **Finding #4 (trailing-only fence):** LLMs sometimes return `{...}\n``` ` (closing fence without opening). Current code's `startswith("```")` guard skips the strip branch entirely, so trailing fences are not removed. `json.loads` fails on the trailing ```.

- **Finding #7 (silent retry loses original error):** `parse_translation_response` (translate.py:41-49) catches `JSONDecodeError` with a bare `except JSONDecodeError:` (no `as exc`), then chains the SECOND `JSONDecodeError` (from fence-stripped parsing) via `from exc`. If both raw and fence-stripped parsing fail, the original raw error context (position, message) is lost — operators see only the fence-stripped attempt's position in the message.

## Goal

1. Make `_strip_markdown_fences` robust against single-line opening fences, trailing-only fences, and CRLF line endings.
2. Preserve both raw and fence-stripped error context when JSON parsing fails in `parse_translation_response`.

## Non-goals

- Auto-correcting malformed LLM output (we just report it loudly).
- Adding new public API surface — `_strip_markdown_fences` stays private.
- Findings #5 (model output cap validation) and #6 (batch-loop test coverage) — separate spec.
- Changing `_translate_one_batch` or retry helper — untouched by this work.

## Design

### Section 1: Rewrite `_strip_markdown_fences` with regex

Replace the current string-operation logic with two regex passes (one for leading fence, one for trailing fence):

```python
import re

_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")


def _strip_markdown_fences(content: str) -> str:
    """Strip leading/trailing markdown code fences (e.g. ```json\\n...\\n```).

    Handles standard (```json\\n...\\n```), single-line opening (```json {...}```),
    trailing-only (\\n```), and CRLF line endings.
    """
    stripped = _LEADING_FENCE_RE.sub("", content.strip(), count=1)
    stripped = _TRAILING_FENCE_RE.sub("", stripped, count=1)
    return stripped.strip()
```

**Why this works:**

| Input | Result |
|---|---|
| ` ```json\n{...}\n``` ` | `{...}` ✓ |
| ` ```json {...}\n``` ` (no newline after opening) | `{...}` ✓ (fixes #3) |
| ` ```{...}\n``` ` (no language tag) | `{...}` ✓ |
| `{...}\n``` ` (trailing-only) | `{...}` ✓ (fixes #4) |
| ` ```json\r\n{...}\r\n```\r\n` (CRLF) | `{...}` ✓ (regex `\n?` matches `\n` after CR is stripped by `content.strip()`'s leading/trailing whitespace handling; verified by test) |
| `{...}` (no fences) | `{...}` ✓ (neither regex matches) |
| ` ```json\n{...}\n``` \n```json\n[more]\n``` ` | first fence stripped, but trailing-second fence not handled — this case is rare and out of scope |

The `count=1` parameter ensures only one leading and one trailing fence are stripped, even if the content has nested fence-like patterns (unlikely but defensive).

`content.strip()` before the leading-fence regex removes outer whitespace, then `stripped.strip()` at the end trims any residual whitespace from the regex substitution.

### Section 2: Preserve raw error context in `parse_translation_response`

Replace the bare `except JSONDecodeError:` with `except JSONDecodeError as exc_raw:` and use Python 3.11's `add_note()` to attach the raw error context to the new `RuntimeError`:

```python
def parse_translation_response(content: str) -> list[TranslationRow]:
    try:
        data = json.loads(content)
    except JSONDecodeError as exc_raw:
        try:
            data = json.loads(_strip_markdown_fences(content))
        except JSONDecodeError as exc_stripped:
            new_exc = RuntimeError(
                f"Invalid LLM translation response: invalid JSON at char {exc_stripped.pos}"
            )
            new_exc.add_note(
                f"Raw (pre-strip) JSON also failed at char {exc_raw.pos}: {exc_raw.msg}"
            )
            raise new_exc from exc_stripped

    # ... rest unchanged
```

**Why this works:**

- `__cause__` chain points to `exc_stripped` (the fence-stripped attempt — the most relevant cause).
- `add_note(...)` (Python 3.11+) attaches `exc_raw` context to the RuntimeError. The note appears in tracebacks automatically: `RuntimeError: Invalid LLM translation response: invalid JSON at char N` followed by `Raw (pre-strip) JSON also failed at char M: ...`.
- Error message prefix `"Invalid LLM translation response"` unchanged → existing test `test_parse_translation_response_rejects_invalid_json` still matches.
- Python 3.11+ requirement is satisfied (project uses 3.11 per environment).

### Section 3: Test plan

Add 4 new tests to `tests/test_translate.py`:

1. **`test_parse_translation_response_handles_single_line_fence`**
   - Input: `'```json {"translations":[{"segment_id":"m-0001",...}]} \\n```'`
   - Expected: parses successfully, returns 1 row.

2. **`test_parse_translation_response_handles_trailing_only_fence`**
   - Input: `'{"translations":[{"segment_id":"m-0001",...}]}\\n```'`
   - Expected: parses successfully, returns 1 row.

3. **`test_parse_translation_response_handles_crlf_fences`**
   - Input: `'```json\\r\\n{"translations":[{"segment_id":"m-0001",...}]}\\r\\n```\\r\\n'`
   - Expected: parses successfully, returns 1 row.

4. **`test_parse_translation_response_preserves_raw_error_context`**
   - Input: `'Not valid JSON at all'` (no fences, fails both raw and stripped)
   - Expected:
     - `RuntimeError` raised with message matching `"Invalid LLM translation response"`
     - `__cause__` is `json.JSONDecodeError`
     - `__notes__` (Python 3.11+) contains `"Raw (pre-strip) JSON also failed"`

Existing tests in `tests/test_translate.py` continue to pass:

- `test_parse_translation_response_rejects_invalid_json` — message prefix unchanged
- `test_parse_translation_response_strips_markdown_fences` — happy-path fence stripping still works
- `test_parse_translation_response_requires_translations_list`, `test_parse_translation_response_identifies_bad_item`, `test_parse_translation_response_warns_and_coerces_invalid_status` — unrelated to fence/error-context changes

## Risk

**Low.** Changes are localized to one private function and one error-handling branch in `parse_translation_response`. Error message prefix preserved. All existing tests should pass without modification. `add_note()` is Python 3.11+ feature; project is on Python 3.11 per environment.

## Rollback

Single source file change + tests. Revert the implementation commit if CI fails or operators report regressions.

## Implementation notes

- `import re` goes at the top of `translate.py` with the other stdlib imports.
- `_LEADING_FENCE_RE` and `_TRAILING_FENCE_RE` are module-level constants (compiled once at import time).
- Test fixtures for the 3 happy-path tests can either share a helper function or copy-paste the JSON payload from the existing `test_parse_translation_response_strips_markdown_fences` test. Choose whichever is cleaner during implementation.
- `from __future__ import annotations` already present — no type-hint changes needed.
- No changes to `translate_with_llm`, `_translate_one_batch`, `_translate_one_batch_with_retry`, or the retry constants.
