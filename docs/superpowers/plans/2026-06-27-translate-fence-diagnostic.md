# Translate Fence Parsing + Diagnostic Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `_strip_markdown_fences` robust against single-line opening fences, trailing-only fences, and CRLF line endings; preserve raw JSON error context when both raw and fence-stripped parsing fail.

**Architecture:** Rewrite `_strip_markdown_fences` to use two regex passes (one for leading, one for trailing). Add `add_note()` to the RuntimeError in `parse_translation_response`'s retry branch so the raw exception's position and message appear in tracebacks.

**Tech Stack:** Python 3.11+, pytest, monkeypatch, `re` stdlib, `add_note()` (Python 3.11+ exception feature).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-translate-fence-diagnostic-design.md` (authoritative for behavior).
- Two regex module constants: `_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?")` and `_TRAILING_FENCE_RE = re.compile(r"\n?```$")` — compiled once at module load.
- `count=1` on both `re.sub()` calls — only strip one leading and one trailing fence.
- `_strip_markdown_fences` returns `stripped.strip()` at the end to handle any residual whitespace from regex substitution.
- TDD discipline: write failing test, run to verify failure, implement minimal code, run to verify pass.
- `from __future__ import annotations` already present in `translate.py` (line 1) — `Exception | None` syntax works.
- Python 3.11+ required (project already on 3.11 per environment). `add_note()` is the standard mechanism for attaching extra context to exceptions.
- Existing tests in `tests/test_translate.py` must continue to pass without modification (`test_parse_translation_response_rejects_invalid_json`, `test_parse_translation_response_strips_markdown_fences`, etc.).
- No changes to `translate_with_llm`, `_translate_one_batch`, `_translate_one_batch_with_retry`, or any retry constants.

---

### Task 1: Rewrite `_strip_markdown_fences` with regex

**Files:**
- Modify: `src/vietdub/translate.py:1-7` (add `import re`) and `src/vietdub/translate.py:29-38` (replace function)
- Modify: `tests/test_translate.py` (add 3 tests)

**Interfaces:**
- Consumes: `parse_translation_response` (existing).
- Produces: `_strip_markdown_fences(content: str) -> str` (rewritten to handle 4 fence formats).

- [ ] **Step 1: Add 3 tests**

Add the following helper and 3 tests to `tests/test_translate.py` (place them near the existing `test_parse_translation_response_strips_markdown_fences`):

```python
def _valid_translations_payload():
    return {
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
            }
        ]
    }


def test_parse_translation_response_handles_single_line_fence():
    payload = json.dumps(_valid_translations_payload())
    # No newline between ```json and {
    fenced = "```json " + payload + "\n```"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_handles_trailing_only_fence():
    payload = json.dumps(_valid_translations_payload())
    # No opening fence, only trailing ```
    fenced = payload + "\n```"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_handles_crlf_fences():
    payload = json.dumps(_valid_translations_payload())
    fenced = "```json\r\n" + payload + "\r\n```\r\n"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"
```

- [ ] **Step 2: Run new tests, verify expected failures**

Run: `pytest tests/test_translate.py::test_parse_translation_response_handles_single_line_fence tests/test_translate.py::test_parse_translation_response_handles_trailing_only_fence tests/test_translate.py::test_parse_translation_response_handles_crlf_fences -v`

Expected:
- `test_parse_translation_response_handles_single_line_fence`: **FAIL** — current logic's `find("\n")` returns the position after the JSON content, slicing keeps only ` ``` `, `rfind("```")` strips it, function returns empty string, `json.loads("")` raises, `parse_translation_response` raises `RuntimeError` instead of returning rows.
- `test_parse_translation_response_handles_trailing_only_fence`: **FAIL** — current logic's `startswith(" ``` ")` is False, the strip branch is skipped, trailing ` ``` ` is not removed, `json.loads` raises, `parse_translation_response` raises `RuntimeError` instead of returning rows.
- `test_parse_translation_response_handles_crlf_fences`: **PASS** — current logic happens to handle CRLF correctly because `strip()` removes the trailing `\r\n` and `find("\n")` matches the `\n` part of `\r\n`. This test serves as a regression guard.

- [ ] **Step 3: Add `import re` and rewrite `_strip_markdown_fences`**

In `src/vietdub/translate.py`, after the existing `import json` line (line 4), add:
```python
import re
```

Replace the existing `_strip_markdown_fences` function (currently lines 29-38) with:

```python
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

- [ ] **Step 4: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_parse_translation_response_handles_single_line_fence tests/test_translate.py::test_parse_translation_response_handles_trailing_only_fence tests/test_translate.py::test_parse_translation_response_handles_crlf_fences -v`

Expected: all 3 pass.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest -v`

Expected: 117/117 pass (114 prior + 3 new = 117). The existing happy-path test `test_parse_translation_response_strips_markdown_fences` (which uses the standard ` ```json\n{...}\n``` ` format) continues to pass — the new regex-based implementation handles that format correctly.

- [ ] **Step 6: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): robust fence stripping for LLM responses

Replace string-operation logic with two regex passes that handle
single-line opening fences (\\\`\\\`\\\`json {...}\\\`\\\`\\\`), trailing-only
fences ({...}\\\n\\\`\\\`\\\`), and CRLF line endings."
```

---

### Task 2: Preserve raw error context in `parse_translation_response`

**Files:**
- Modify: `src/vietdub/translate.py:42-49` (the `JSONDecodeError` retry block)
- Modify: `tests/test_translate.py` (add 1 test)

**Interfaces:**
- Consumes: existing `parse_translation_response` and `JSONDecodeError`.
- Produces: `RuntimeError` with `__cause__` = fence-stripped exception, `__notes__` contains raw exception's position and message.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_translate.py` (near the other `test_parse_translation_response_*` tests):

```python
def test_parse_translation_response_preserves_raw_error_context():
    with pytest.raises(RuntimeError, match="Invalid LLM translation response") as exc_info:
        parse_translation_response("Not valid JSON at all")

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("Raw (pre-strip) JSON also failed" in n for n in notes)
```

- [ ] **Step 2: Run new test, verify it fails**

Run: `pytest tests/test_translate.py::test_parse_translation_response_preserves_raw_error_context -v`

Expected: **FAIL** — current code's `RuntimeError` has no `__notes__` attribute set. The `getattr(exc_info.value, "__notes__", [])` returns `[]`, so the `any(...)` assertion fails.

- [ ] **Step 3: Update `parse_translation_response` to add raw error context**

In `src/vietdub/translate.py`, replace the current retry block (currently lines 42-49):

```python
    try:
        data = json.loads(content)
    except JSONDecodeError:
        # Retry after stripping markdown code fences (LLMs sometimes wrap JSON in ```json ... ```)
        try:
            data = json.loads(_strip_markdown_fences(content))
        except JSONDecodeError as exc:
            raise RuntimeError(f"Invalid LLM translation response: invalid JSON at char {exc.pos}") from exc
```

with:

```python
    try:
        data = json.loads(content)
    except JSONDecodeError as exc_raw:
        # Retry after stripping markdown code fences (LLMs sometimes wrap JSON in ```json ... ```)
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
```

- [ ] **Step 4: Run new test, verify it passes**

Run: `pytest tests/test_translate.py::test_parse_translation_response_preserves_raw_error_context -v`

Expected: PASS.

- [ ] **Step 5: Run existing tests that exercise this code path**

Run: `pytest tests/test_translate.py::test_parse_translation_response_rejects_invalid_json tests/test_translate.py::test_parse_translation_response_requires_translations_list tests/test_translate.py::test_parse_translation_response_identifies_bad_item -v`

Expected: all pass. The `RuntimeError` message prefix `"Invalid LLM translation response"` is unchanged in all error paths. The `match` patterns in existing tests (`"Invalid LLM translation response"`, `"translations"`, `"translation item 1"`) still match.

- [ ] **Step 6: Run full suite**

Run: `pytest -v`

Expected: 118/118 pass (117 prior + 1 new = 118).

- [ ] **Step 7: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): preserve raw error context in JSON parsing

When both raw and fence-stripped JSON parsing fail, attach the raw
exception's position and message to the RuntimeError via Python 3.11's
add_note(). The fence-stripped exception remains the __cause__ for
the traceback chain."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Two regex constants + new `_strip_markdown_fences` body | Task 1 |
| `import re` added | Task 1 |
| 3 new tests (single-line, trailing-only, CRLF) | Task 1 |
| Existing tests still pass | Task 1 (implicit in Step 5) |
| `parse_translation_response` captures both exceptions | Task 2 |
| `add_note()` for raw context | Task 2 |
| `__cause__` chains to fence-stripped exception | Task 2 |
| 1 new test (error context) | Task 2 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later", "fill in details".
- No "Add appropriate error handling" without specific code.
- No "Similar to Task N" — code is repeated where needed.
- Every step with code change shows the code.
- Every test has full assertions.
- Every command has expected output.

**3. Type consistency:**

- `_strip_markdown_fences(content: str) -> str` signature unchanged.
- `parse_translation_response(content: str) -> list[TranslationRow]` signature unchanged.
- `_valid_translations_payload()` helper: no signature (returns `dict`).
- `_LEADING_FENCE_RE` and `_TRAILING_FENCE_RE`: module-level constants, type `re.Pattern[str]`.

**4. Test/impl ordering:**

- Task 1 Step 2 explicitly notes that one test (CRLF) passes immediately as a regression guard, not a TDD red-green test. This is acceptable because the spec identifies CRLF handling as part of the rewrite goal even though the current implementation happens to handle it.
- Task 2 follows strict TDD: write failing test → verify failure → implement → verify pass.
- Both tasks end with `pytest -v` showing full suite green before commit.
