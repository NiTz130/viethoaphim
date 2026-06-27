# Translation Length Matching Post-Processing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add post-processing length check inside `parse_translation_response` that warns (no fail, no retry) when Vietnamese translation length falls outside ±20% of `target_vi_chars`.

**Architecture:** Single source-file change to `src/vietdub/translate.py`. Add a module-level `_LENGTH_TOLERANCE = 0.20` constant and a private `_warn_oversized_translations(rows)` helper. Call the helper at the end of `parse_translation_response` (before `return rows`). Reuse the existing `_length_target_vi_chars(start_ms, end_ms)` function from Spec 1 to compute targets — no duplicate logic. Add 5 tests covering oversized / undersized / within tolerance / empty / floor region.

**Tech Stack:** Python 3.11+, pytest, `capsys` fixture.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-length-matching-post-processing-design.md` (authoritative for behavior).
- `_LENGTH_TOLERANCE = 0.20` (module-level constant in `src/vietdub/translate.py`).
- `_warn_oversized_translations(rows: list[TranslationRow]) -> None` (private helper).
- Skip conditions: empty `text_vi` AND `target < 10` (floor region).
- Tolerance check: `len(text_vi) > target * 1.20` → warn over; `len(text_vi) < target * 0.80` → warn under.
- All warnings go to `sys.stderr` with `[LEN WARNING]` prefix.
- Reuse `_length_target_vi_chars` from Spec 1 — no duplicate logic.
- No changes to: `build_translation_prompt`, `_translate_one_batch`, `_translate_one_batch_with_retry`, `translate_with_llm`, batching, retry, `_EXAMPLES`, `KNOWN_MODEL_OUTPUT_CAPS`, retry constants.
- Existing 130 tests must continue to pass.
- TDD discipline: write failing tests, verify fail, implement, verify pass.

---

### Task 1: Add length check helper + wire into parse_translation_response

**Files:**
- Modify: `src/vietdub/translate.py` (add `_LENGTH_TOLERANCE` constant near existing constants; add `_warn_oversized_translations` helper; call helper in `parse_translation_response` before `return rows`)
- Modify: `tests/test_translate.py` (add 5 tests at the end)

**Interfaces:**
- Consumes: existing `TranslationRow`, `parse_translation_response`, `_length_target_vi_chars`.
- Produces: `parse_translation_response` emits `[LEN WARNING]` lines to `sys.stderr` when translations exceed ±20% of target. Return value unchanged.

- [ ] **Step 1: Add 5 failing tests**

Add the following to `tests/test_translate.py` at the end of the file:

```python
def test_parse_translation_response_warns_on_oversized_translation(capsys):
    """Translation > 120% of target emits [LEN WARNING] to stderr."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0042",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": None,
            "text_cn": "你好",
            "text_vi": "啊" * 50,
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    captured = capsys.readouterr()
    assert "[LEN WARNING]" in captured.err
    assert "m-0042" in captured.err
    assert "over" in captured.err


def test_parse_translation_response_warns_on_undersized_translation(capsys):
    """Translation < 80% of target emits [LEN WARNING] to stderr."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0099",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": None,
            "text_cn": "你好世界",
            "text_vi": "xin chào",
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    captured = capsys.readouterr()
    assert "[LEN WARNING]" in captured.err
    assert "m-0099" in captured.err
    assert "under" in captured.err


def test_parse_translation_response_no_warn_within_tolerance(capsys):
    """Translation within ±20% of target emits no [LEN WARNING]."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0001",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": None,
            "text_cn": "你好",
            "text_vi": "xin chào bạn",
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    captured = capsys.readouterr()
    assert "[LEN WARNING]" not in captured.err


def test_parse_translation_response_skips_length_check_for_empty_translation(capsys):
    """Empty text_vi is not length-checked (would always 'under')."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-empty",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": None,
            "text_cn": "你好",
            "text_vi": "",
            "context_note": "",
            "status": "draft",
        }]
    })
    rows = parse_translation_response(payload)
    assert len(rows) == 1
    assert rows[0].text_vi == ""
    captured = capsys.readouterr()
    assert "[LEN WARNING]" not in captured.err


def test_parse_translation_response_skips_length_check_for_very_short_segments(capsys):
    """Segments with target < 10 chars (floor region) skip length check."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-short",
            "start_ms": 0,
            "end_ms": 100,
            "speaker": None,
            "text_cn": "hi",
            "text_vi": "a" * 50,
            "context_note": "",
            "status": "draft",
        }]
    })
    rows = parse_translation_response(payload)
    assert len(rows) == 1
    captured = capsys.readouterr()
    assert "[LEN WARNING]" not in captured.err
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_translate.py::test_parse_translation_response_warns_on_oversized_translation tests/test_translate.py::test_parse_translation_response_warns_on_undersized_translation tests/test_translate.py::test_parse_translation_response_no_warn_within_tolerance tests/test_translate.py::test_parse_translation_response_skips_length_check_for_empty_translation tests/test_translate.py::test_parse_translation_response_skips_length_check_for_very_short_segments -v`

Expected: all 5 fail (the helper doesn't exist yet → ImportError on `_warn_oversized_translations`, or `parse_translation_response` doesn't emit warnings).

- [ ] **Step 3: Add `_LENGTH_TOLERANCE` constant and `_warn_oversized_translations` helper**

In `src/vietdub/translate.py`, after the existing `_length_target_vi_chars` function (around line 60), add:

```python
_LENGTH_TOLERANCE = 0.20  # ±20% of target_vi_chars


def _warn_oversized_translations(rows: list[TranslationRow]) -> None:
    """Warn if Vietnamese translation length is outside ±20% of target.

    Skips empty translations and segments in the floor region (target < 10).
    Warnings go to stderr with [LEN WARNING] prefix for easy filtering.
    """
    for row in rows:
        text_len = len(row.text_vi)
        if text_len == 0:
            continue
        target = _length_target_vi_chars(row.start_ms, row.end_ms)
        if target < 10:
            continue
        if text_len > target * (1 + _LENGTH_TOLERANCE):
            over_pct = int((text_len / target - 1) * 100)
            print(
                f"[LEN WARNING] segment {row.segment_id}: {text_len} chars vs target {target} "
                f"({over_pct}% over). Vietnamese translation may exceed dubbing timing.",
                file=sys.stderr,
            )
        elif text_len < target * (1 - _LENGTH_TOLERANCE):
            under_pct = int((1 - text_len / target) * 100)
            print(
                f"[LEN WARNING] segment {row.segment_id}: {text_len} chars vs target {target} "
                f"({under_pct}% under). Vietnamese translation may read too fast for dubbing.",
                file=sys.stderr,
            )
```

- [ ] **Step 4: Wire helper into `parse_translation_response`**

In `src/vietdub/translate.py`, find `parse_translation_response` (around lines 80-86) and insert the helper call between the rows loop and the `return rows` statement:

After (currently):
```python
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid LLM translation response: translation item {index} "
                f"(segment_id={item.get('segment_id', '?')!r}) failed validation: {exc}"
            ) from exc
    return rows
```

Change to:
```python
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid LLM translation response: translation item {index} "
                f"(segment_id={item.get('segment_id', '?')!r}) failed validation: {exc}"
            ) from exc

    _warn_oversized_translations(rows)

    return rows
```

- [ ] **Step 5: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_parse_translation_response_warns_on_oversized_translation tests/test_translate.py::test_parse_translation_response_warns_on_undersized_translation tests/test_translate.py::test_parse_translation_response_no_warn_within_tolerance tests/test_translate.py::test_parse_translation_response_skips_length_check_for_empty_translation tests/test_translate.py::test_parse_translation_response_skips_length_check_for_very_short_segments -v`

Expected: all 5 pass.

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `pytest -v`

Expected: 135/135 pass (was 130 + 5 new = 135). Existing tests use translations that fit within ±20% of target (e.g., "Xin chào" for 1s segment ≈ 9 chars vs target ≈ 15, within tolerance). The existing warning about status coercion still works (different `[Warning]` prefix, not affected by `[LEN WARNING]` change).

- [ ] **Step 7: Commit**

```bash
cd C:\code\viethoaphimv4
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): warn when Vietnamese translation length is outside ±20% target

Add _warn_oversized_translations helper called at the end of
parse_translation_response. Warns to stderr (no fail, no retry) when
len(text_vi) falls outside ±20% of target_vi_chars (computed from
segment duration via _length_target_vi_chars from Spec 1).

Skips empty translations and the floor region (target < 10 chars).
Helps operators spot timing-mismatched translations in CSV review."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Add length check in `parse_translation_response` | Task 1 Steps 3, 4 |
| `_LENGTH_TOLERANCE = 0.20` constant | Task 1 Step 3 |
| `_warn_oversized_translations` helper | Task 1 Step 3 |
| Reuse `_length_target_vi_chars` from Spec 1 | Task 1 Step 3 |
| Skip empty `text_vi` | Task 1 Step 3 |
| Skip floor region (`target < 10`) | Task 1 Step 3 |
| Both over and under warnings | Task 1 Step 3 |
| `[LEN WARNING]` stderr prefix | Task 1 Step 3 |
| 5 tests (oversized / undersized / within tolerance / empty / floor) | Task 1 Step 1 |
| No regression in existing 130 tests | Task 1 Step 6 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- All code blocks show full implementation (constant, helper, helper call, all 5 tests).
- Exact commands with expected output throughout.
- Test code is complete (not "similar to..." or "add appropriate assertions").

**3. Type consistency:**

- `_LENGTH_TOLERANCE: float = 0.20` — matches usage in helper.
- `_warn_oversized_translations(rows: list[TranslationRow]) -> None` — signature matches all 5 test usages and the call site.
- `TranslationRow` model unchanged from Spec 1.

**4. TDD discipline:**

- Step 1: write 5 failing tests.
- Step 2: verify all 5 fail (RED).
- Steps 3-4: implement helper + wire call.
- Step 5: verify all 5 pass (GREEN).
- Step 6: full suite regression check.
- Step 7: commit only after green.
