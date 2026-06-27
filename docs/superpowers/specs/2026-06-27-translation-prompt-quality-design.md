# Translation Prompt Quality — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (prompt construction) + `tests/test_translate.py`
**Status:** Approved (design), pending implementation

## Context

The viethoaphimv4 translation pipeline (`src/vietdub/translate.py:build_translation_prompt` and `_translate_one_batch`) uses a 2-prompt structure (system + user), but the prompts are minimal:

**Current system prompt** (1 sentence):
```
You are a Vietnamese localization editor for Chinese comedy cartoons. 
Return JSON only with a top-level "translations" array.
```

**Current user prompt instructions** (6 lines):
- Return JSON only with `"translations"` array
- Each translation must include `segment_id, start_ms, end_ms, speaker, text_cn, text_vi, context_note, status`
- Translate Chinese cartoon dialogue into natural Vietnamese
- Use a silly, meme-friendly tone when the source is comedic
- Keep Vietnamese lines short enough for dubbing timing
- Preserve names and pronouns using the supplied context

These prompts are insufficient for production-quality translations. Issues observed:
- Tone guidance ("silly, meme-friendly") is vague — what concretely counts as "meme-friendly"?
- No few-shot examples — LLM has to infer desired output style from vague instructions
- No length guidance — "keep short" is subjective; LLM often produces Vietnamese much longer than Chinese, breaking dub timing
- No output format examples — LLM occasionally returns bare arrays or wraps in extra braces, causing parse failures

This is Spec 1 of 4 (ordered by complexity): Prompt Quality → Length Matching → Glossary Consistency → Multi-pass Review.

## Goal

Improve translation quality by:
1. Adding 3 few-shot examples in the user prompt (in-context learning for desired style)
2. Refining the system prompt with concrete tone guidance and Vietnamese language conventions
3. Adding per-segment `target_vi_chars` based on duration, computed from Vietnamese speech rate
4. Adding CORRECT/INCORRECT output format examples to reduce parse failures

## Non-goals

- Post-processing validation (length enforcement, retry on mismatch) — Spec 2 (Length Matching)
- Cross-batch glossary/state — Spec 3 (Glossary Consistency)
- Multi-pass review/refine loop — Spec 4 (Multi-pass Review)
- Changes to `translate_with_llm` flow, batching, or retry logic
- Changes to `parse_translation_response`, fence handling, error context (already addressed)
- Model parameter tuning (temperature, top_p) — out of scope for prompt-only work

## Design

### Section 1: Refined system prompt

Replace the current 1-sentence system prompt (translate.py:168-170) with:

```python
system_prompt = (
    "You are a Vietnamese dubbing translator for Chinese comedy cartoons.\n\n"
    "TONE: Use natural, colloquial Vietnamese. Embrace meme culture — common "
    "internet slang like 'vl', 'wtf', 'haha', 'bro' is appropriate for comedic "
    "moments. For serious/dramatic moments, use more formal Vietnamese.\n\n"
    "CONVENTIONS:\n"
    "- Preserve Chinese names in their Vietnamese-accepted form "
    "(e.g., 长孙无忌 → Trưởng Tôn Vô Kỵ, 白素贞 → Bạch Tố Trinh)\n"
    "- Use 'ngươi' (archaic you) for historical/fantasy settings, 'bạn' for modern\n"
    "- Honor 'translation, not transliteration' — convey meaning, not sounds\n\n"
    "Return JSON only with a top-level 'translations' array."
)
```

### Section 2: Few-shot examples in user prompt

Add a static `EXAMPLES` constant and inject it into the user payload:

```python
EXAMPLES: list[dict] = [
    {
        "segment_id": "ex-1",
        "text_cn": "你这个笨蛋！",
        "text_vi": "Mày ngu vậy!",
    },
    {
        "segment_id": "ex-2",
        "text_cn": "长老，我们该怎么办？",
        "text_vi": "Trưởng lão, giờ chúng ta phải làm sao?",
    },
    {
        "segment_id": "ex-3",
        "text_cn": "哈哈哈哈，你真是太有趣了！",
        "text_vi": "Hahaha, mày buồn cười thiệt chứ!",
    },
]
```

Examples are pure reference — they show LLM the desired style but the LLM does NOT need to translate them; they're just context.

### Section 3: Per-segment length targets

Add a helper function and include `target_vi_chars` in each segment dict:

```python
def _length_target_vi_chars(start_ms: int, end_ms: int) -> int:
    """Compute target Vietnamese character count for a segment.
    
    Vietnamese conversational speech is ~14 chars/sec; add 10% buffer for
    punctuation and natural variation. Floor at 10 chars (very short segments).
    """
    duration_s = (end_ms - start_ms) / 1000.0
    return max(10, int(duration_s * 14 * 1.1))
```

In `build_translation_prompt`, augment each segment dict:

```python
"segments": [
    {**segment.model_dump(), "target_vi_chars": _length_target_vi_chars(
        segment.start_ms, segment.end_ms
    )}
    for segment in segments
]
```

Example: 5-second segment → `target_vi_chars = 77`.

### Section 4: Output format examples

Add CORRECT/INCORRECT examples to the instructions block:

```python
"instructions": [
    "Return JSON only with a top-level \"translations\" array.",
    "Each translation must include segment_id, start_ms, end_ms, speaker, text_cn, text_vi, context_note, and status.",
    "Translate Chinese cartoon dialogue into natural Vietnamese.",
    "Use a silly, meme-friendly tone when the source is comedic.",
    f"Aim for target_vi_chars characters in text_vi (Vietnamese ≈ 14 chars/sec + 10% buffer).",
    "Preserve names and pronouns using the supplied context.",
    "OUTPUT FORMAT (CORRECT): {\"translations\": [{\"segment_id\": \"m-0001\", \"text_vi\": \"...\", ...}]}",
    "OUTPUT FORMAT (INCORRECT — do NOT do this):",
    "  [{\"segment_id\": \"m-0001\", ...}]",
    "  {\"translations\": [...]}",
    "  Wrapped in markdown fences or extra braces",
]
```

This explicitly shows the LLM the exact JSON shape expected, reducing parse failures.

### Section 5: Test plan

Add 5 tests to `tests/test_translate.py`:

1. **`test_build_translation_prompt_includes_examples`**
   ```python
   def test_build_translation_prompt_includes_examples():
       prompt_json = build_translation_prompt(
           [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
           context_bundle={},
       )
       payload = json.loads(prompt_json)
       assert "examples" in payload
       assert len(payload["examples"]) >= 3
       for ex in payload["examples"]:
           assert "segment_id" in ex
           assert "text_cn" in ex
           assert "text_vi" in ex
   ```

2. **`test_build_translation_prompt_includes_length_targets`**
   ```python
   def test_build_translation_prompt_includes_length_targets():
       prompt_json = build_translation_prompt(
           [TimedSegment(id="m-0001", start_ms=0, end_ms=5000, text="测试")],
           context_bundle={},
       )
       payload = json.loads(prompt_json)
       seg = payload["segments"][0]
       assert "target_vi_chars" in seg
       assert seg["target_vi_chars"] == 77  # 5s * 14 * 1.1 ≈ 77
   ```

3. **`test_length_target_vi_chars_minimum_floor`**
   ```python
   def test_length_target_vi_chars_minimum_floor():
       prompt_json = build_translation_prompt(
           [TimedSegment(id="m-0001", start_ms=0, end_ms=100, text="hi")],
           context_bundle={},
       )
       payload = json.loads(prompt_json)
       seg = payload["segments"][0]
       assert seg["target_vi_chars"] >= 10  # Floor at 10 chars
   ```

4. **`test_translation_examples_have_valid_format`**
   ```python
   def test_translation_examples_have_valid_format():
       prompt_json = build_translation_prompt(
           [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
           context_bundle={},
       )
       payload = json.loads(prompt_json)
       for ex in payload["examples"]:
           assert ex["segment_id"].startswith("ex-")
           assert ex["text_cn"]
           assert ex["text_vi"]
   ```

5. **`test_translate_pipeline_produces_valid_response`**
   ```python
   def test_translate_pipeline_produces_valid_response(monkeypatch):
       """Integration test: verify the new prompts don't break existing parse path."""
       # Use existing _FakeMessages pattern; just verify prompt still parses
       monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))
       rows = translate_with_llm(
           segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
           context_bundle={},
           settings=_settings(),
       )
       assert len(rows) == 1
       assert rows[0].text_vi == "Xin chào"  # from _FakeMessages
   ```

## Risk

**Medium.** Improved prompts will produce different translations, but should NOT change the API contract or break existing tests. The risk is that the LLM behavior changes subtly. Mitigation: spec is bounded to prompt changes only, no API changes. Existing test suite (125 tests) covers the full pipeline — if it still passes, the contract is intact.

## Rollback

Single source file change + tests. Revert the implementation commit if CI fails or operators report regression.

## Implementation notes

- `EXAMPLES` is a module-level constant (compiled once at import).
- `_length_target_vi_chars` is a private helper (leading underscore).
- System prompt change is in `_translate_one_batch` (line 168-170 area).
- User prompt change is in `build_translation_prompt` (line 89-102 area).
- All changes are localized; no impact on retry, batching, parsing, or error handling.
- Existing prompt size grows by ~500 bytes (few-shot examples + format examples + length targets). Still well under `max_tokens=8192`.
