# Translation Length Matching Post-Processing — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (parse_translation_response) + `tests/test_translate.py`
**Status:** Approved (design), pending implementation

## Context

Spec 1 (commit `0cadca4`) added `target_vi_chars` per segment to the translation prompt, computed from segment duration via Vietnamese speech rate (~14 chars/sec, floor 10 chars). This tells the LLM the target length but does not enforce it — Vietnamese translations may still exceed the target.

For dubbing, segments where Vietnamese text significantly exceeds the Chinese source duration cause problems:
- TTS audio doesn't fit the original video's lip-sync window
- Operator has to manually shorten translations during review
- Hard to spot in a 200+ segment job without automated checking

This is Spec 2 of 4 (Prompt → Length → Glossary → Multi-pass Review).

## Goal

Add post-processing length check inside `parse_translation_response` that warns (without blocking) when Vietnamese translation length falls outside ±20% of `target_vi_chars`.

## Non-goals

- Auto-retry with stricter prompt — Spec 4 (Multi-pass Review)
- Auto-truncate — operator should review and edit, not have translations silently cut
- Strict-fail mode — would block jobs over the tolerance threshold
- Undersized threshold tuning — fixed at ±20%
- Cross-batch consistency (current spec is per-segment only) — could be a follow-up
- CSV column for length report — operator can read stderr; CSV stays simple

## Design

### Section 1: Length check inside `parse_translation_response`

Add the length check after the rows list is built (line 75 of translate.py), before the `return rows` statement:

```python
    rows: list[TranslationRow] = []
    for index, item in enumerate(translations, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid LLM translation response: translation item {index} is not an object")
        if item.get("status") not in ALLOWED_REVIEW_STATUSES:
            print(
                f"Warning: LLM returned invalid status {item.get('status')!r} "
                f"for segment {item.get('segment_id')!r}; coercing to 'draft'",
                file=sys.stderr,
            )
            item = {**item, "status": "draft"}
        try:
            rows.append(TranslationRow.model_validate(item))
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid LLM translation response: translation item {index} "
                f"(segment_id={item.get('segment_id', '?')!r}) failed validation: {exc}"
            ) from exc

    _warn_oversized_translations(rows)  # NEW

    return rows
```

The check is a separate helper for testability:

```python
_LENGTH_TOLERANCE = 0.20  # ±20%


def _warn_oversized_translations(rows: list[TranslationRow]) -> None:
    """Warn if Vietnamese translation length is outside ±20% of target."""
    for row in rows:
        text_len = len(row.text_vi)
        if text_len == 0:
            continue  # Skip empty translations
        target = _length_target_vi_chars(row.start_ms, row.end_ms)
        if target < 10:
            continue  # Skip the floor region (very short segments)
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

`_LENGTH_TOLERANCE` is a module-level constant for tunability.

### Section 2: Reuse `_length_target_vi_chars` from Spec 1

The same `_length_target_vi_chars(start_ms, end_ms) -> int` function added in Spec 1 (`src/vietdub/translate.py:53-60`) computes the target from row's start_ms/end_ms. Spec 2 reuses it directly — no duplication.

### Section 3: Test plan

Add 5 tests to `tests/test_translate.py`:

1. **`test_parse_translation_response_warns_on_oversized_translation`**
   ```python
   def test_parse_translation_response_warns_on_oversized_translation(capsys):
       """Translation > 120% of target emits [LEN WARNING] to stderr."""
       # Segment duration 1s → target ≈ 15 chars; use 50 chars (way over)
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
   ```

2. **`test_parse_translation_response_warns_on_undersized_translation`**
   ```python
   def test_parse_translation_response_warns_on_undersized_translation(capsys):
       """Translation < 80% of target emits [LEN WARNING] to stderr."""
       # Segment 5s → target ≈ 77 chars; use 5 chars (way under)
       payload = json.dumps({
           "translations": [{
               "segment_id": "m-0099",
               "start_ms": 0,
               "end_ms": 5000,
               "speaker": None,
               "text_cn": "你好世界",
               "text_vi": "xin chào",  # 9 chars, target ≈ 77
               "context_note": "",
               "status": "draft",
           }]
       })
       parse_translation_response(payload)
       captured = capsys.readouterr()
       assert "[LEN WARNING]" in captured.err
       assert "m-0099" in captured.err
       assert "under" in captured.err
   ```

3. **`test_parse_translation_response_no_warn_within_tolerance`**
   ```python
   def test_parse_translation_response_no_warn_within_tolerance(capsys):
       """Translation within ±20% of target emits no [LEN WARNING]."""
       # Segment 1s → target ≈ 15; use 16 chars (within tolerance)
       payload = json.dumps({
           "translations": [{
               "segment_id": "m-0001",
               "start_ms": 0,
               "end_ms": 1000,
               "speaker": None,
               "text_cn": "你好",
               "text_vi": "xin chào bạn",  # 15 chars
               "context_note": "",
               "status": "draft",
           }]
       })
       parse_translation_response(payload)
       captured = capsys.readouterr()
       assert "[LEN WARNING]" not in captured.err
   ```

4. **`test_parse_translation_response_skips_length_check_for_empty_translation`**
   ```python
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
   ```

5. **`test_parse_translation_response_skips_length_check_for_very_short_segments`**
   ```python
   def test_parse_translation_response_skips_length_check_for_very_short_segments(capsys):
       """Segments with target < 10 chars (floor region) skip length check."""
       # 100ms duration → target floored at 10 chars; even 50 chars is over floor
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

## Risk

**Low.** Behavior change is additive (only stderr output). Existing 130 tests should continue to pass — they use translations that are within tolerance (e.g., the existing happy-path "Xin chào" for 1s segment ≈ 9 chars, target ≈ 15, well within tolerance).

## Rollback

Single source file change + tests. Revert the implementation commit if operators find the warnings too noisy.

## Implementation notes

- `_LENGTH_TOLERANCE` is a module-level constant (tunable in one place).
- `_warn_oversized_translations` is a private helper (leading underscore).
- The check runs after row validation, so invalid translations don't trigger length warnings.
- All warnings go to `sys.stderr` — consistent with existing status coercion warnings in `parse_translation_response` (line 73-77) and `translate_with_llm` (line 181-185).
- No changes to `translate_with_llm`, batching, retry, `build_translation_prompt`, `_translate_one_batch`, `_translate_one_batch_with_retry`, `_EXAMPLES`, or any constants.
