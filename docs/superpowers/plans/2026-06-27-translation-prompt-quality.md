# Translation Prompt Quality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve viethoaphimv4 translation quality by refining the system prompt, adding few-shot examples, per-segment length targets, and output format examples.

**Architecture:** Single source-file change to `src/vietdub/translate.py` (modify `build_translation_prompt` + the system prompt in `_translate_one_batch`). Add a module-level `EXAMPLES` constant and a `_length_target_vi_chars` helper. Add 5 tests to `tests/test_translate.py` covering each new behavior.

**Tech Stack:** Python 3.11+, pytest, JSON.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-translation-prompt-quality-design.md` (authoritative)
- `EXAMPLES` constant: 3 entries with `segment_id` (prefixed `ex-`), `text_cn`, `text_vi`.
- `_length_target_vi_chars(start_ms, end_ms) -> int`: `max(10, int(duration_s * 14 * 1.1))`.
- Each segment dict in the user prompt payload gains a `target_vi_chars` field.
- System prompt in `_translate_one_batch` (lines 168-170) becomes multi-paragraph with TONE, CONVENTIONS, JSON output.
- Instructions array in `build_translation_prompt` adds: length-target line, output format CORRECT/INCORRECT examples.
- No changes to: `parse_translation_response`, fence handling, retry, batching, parsing, error handling, validation, `KNOWN_MODEL_OUTPUT_CAPS`, retry constants.
- Existing 125 tests must continue to pass (no behavior changes to existing paths).
- Prompt size grows by ~500 bytes; still well under `max_tokens=8192`.

---

### Task 1: Refine system prompt + add EXAMPLES + length targets + format examples

**Files:**
- Modify: `src/vietdub/translate.py` (refine system prompt around line 168-170; modify `build_translation_prompt` around lines 89-102; add `EXAMPLES` constant + `_length_target_vi_chars` helper)
- Modify: `tests/test_translate.py` (add 5 tests)

**Interfaces:**
- Consumes: existing `translate_with_llm`, `_translate_one_batch`, `_FakeMessages` mock pattern.
- Produces: enhanced `build_translation_prompt` payload with `examples` key + `target_vi_chars` per segment + refined system prompt + format examples.

- [ ] **Step 1: Add 5 failing tests**

Add the following to `tests/test_translate.py` at the end of the file:

```python
def test_build_translation_prompt_includes_examples():
    """build_translation_prompt payload contains 3+ few-shot examples."""
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


def test_build_translation_prompt_includes_length_targets():
    """Each segment dict has target_vi_chars based on duration."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=5000, text="测试")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    seg = payload["segments"][0]
    assert "target_vi_chars" in seg
    assert seg["target_vi_chars"] == 77  # 5s * 14 chars/sec * 1.1 buffer ≈ 77


def test_length_target_vi_chars_minimum_floor():
    """Very short segments still get a minimum target of 10 chars."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=100, text="hi")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    seg = payload["segments"][0]
    assert seg["target_vi_chars"] >= 10


def test_translation_examples_have_valid_format():
    """All examples have segment_id prefix 'ex-' and non-empty text fields."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    for ex in payload["examples"]:
        assert ex["segment_id"].startswith("ex-")
        assert ex["text_cn"]
        assert ex["text_vi"]


def test_translate_pipeline_produces_valid_response():
    """Integration: new prompts don't break the existing parse path."""
    from vietdub.translate import translate_with_llm
    from vietdub.models import TimedSegment
    from tests.test_translate_responses import _FakeAnthropic, _settings
    import sys
    import types

    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
        context_bundle={},
        settings=_settings(),
    )

    assert len(rows) == 1
    assert rows[0].text_vi == "Xin chào"  # from _FakeMessages
    monkey.undo()
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_translate.py::test_build_translation_prompt_includes_examples tests/test_translate.py::test_build_translation_prompt_includes_length_targets tests/test_translate.py::test_length_target_vi_chars_minimum_floor tests/test_translate.py::test_translation_examples_have_valid_format tests/test_translate.py::test_translate_pipeline_produces_valid_response -v`

Expected: all 5 fail (the prompt currently lacks `examples`, `target_vi_chars`, etc.).

- [ ] **Step 3: Add `EXAMPLES` constant and `_length_target_vi_chars` helper to `translate.py`**

In `src/vietdub/translate.py`, after the existing `_TRAILING_FENCE_RE = re.compile(...)` line (around line 31), add:

```python
_EXAMPLES: list[dict] = [
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


def _length_target_vi_chars(start_ms: int, end_ms: int) -> int:
    """Compute target Vietnamese character count for a segment.

    Vietnamese conversational speech is ~14 chars/sec; add 10% buffer for
    punctuation and natural variation. Floor at 10 chars for very short segments.
    """
    duration_s = (end_ms - start_ms) / 1000.0
    return max(10, int(duration_s * 14 * 1.1))
```

- [ ] **Step 4: Update `build_translation_prompt` to include examples, length targets, and format examples**

In `src/vietdub/translate.py`, replace the existing `build_translation_prompt` function (lines 89-102) with:

```python
def build_translation_prompt(segments: list[TimedSegment], context_bundle: dict) -> str:
    payload = {
        "instructions": [
            "Return JSON only with a top-level \"translations\" array.",
            "Each translation must include segment_id, start_ms, end_ms, speaker, text_cn, text_vi, context_note, and status.",
            "Translate Chinese cartoon dialogue into natural Vietnamese.",
            "Use a silly, meme-friendly tone when the source is comedic.",
            "Aim for the target_vi_chars characters shown per segment (Vietnamese ≈ 14 chars/sec + 10% buffer).",
            "Preserve names and pronouns using the supplied context.",
            "OUTPUT FORMAT (CORRECT): {\"translations\": [{\"segment_id\": \"m-0001\", \"text_vi\": \"...\", ...}]}",
            "OUTPUT FORMAT (INCORRECT — do NOT do this):",
            "  [{\"segment_id\": \"m-0001\", ...}]  (bare array, missing top-level object)",
            "  Wrapped in markdown fences or extra braces",
        ],
        "examples": _EXAMPLES,
        "context": context_bundle,
        "segments": [
            {**segment.model_dump(), "target_vi_chars": _length_target_vi_chars(segment.start_ms, segment.end_ms)}
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
```

- [ ] **Step 5: Refine the system prompt in `_translate_one_batch`**

In `src/vietdub/translate.py`, replace the existing system_prompt assignment in `_translate_one_batch` (lines 168-170) with:

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

- [ ] **Step 6: Run new tests, verify all pass**

Run: `pytest tests/test_translate.py::test_build_translation_prompt_includes_examples tests/test_translate.py::test_build_translation_prompt_includes_length_targets tests/test_translate.py::test_length_target_vi_chars_minimum_floor tests/test_translate.py::test_translation_examples_have_valid_format tests/test_translate.py::test_translate_pipeline_produces_valid_response -v`

Expected: all 5 pass.

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `pytest -v`

Expected: 130/130 pass (was 125 + 5 new = 130). The existing happy-path tests use `_FakeAnthropic` and don't care about prompt contents — they only check that the LLM mock's response is parsed correctly. The new prompt changes don't affect the parsing path.

- [ ] **Step 8: Commit**

```bash
cd C:\code\viethoaphimv4
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat(translate): improve translation prompts with examples, tone, length targets

- Add 3 few-shot examples (comedic, dramatic, fantasy) to user prompt
- Refine system prompt with concrete tone guidance and Vietnamese
  language conventions (meme slang, name preservation, formality)
- Add per-segment target_vi_chars computed from duration
  (Vietnamese ~14 chars/sec + 10% buffer, floor 10 chars)
- Add CORRECT/INCORRECT output format examples to reduce parse failures

Output of translate_with_llm is unchanged (existing parse path), but
translations should now be more colloquial, length-matched, and
consistent in style across batches."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Refined system prompt (Section 1) | Task 1 Step 5 |
| Few-shot examples (Section 2) | Task 1 Steps 3, 4 |
| Per-segment length targets (Section 3) | Task 1 Steps 3, 4 |
| Output format examples (Section 4) | Task 1 Step 4 |
| 5 tests (Section 5) | Task 1 Step 1 |
| No regression in 125 existing tests | Task 1 Step 7 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- Every code block is full code (EXAMPLES, _length_target_vi_chars, build_translation_prompt, system_prompt, all 5 tests).
- Exact commands with expected output throughout.
- No references to undefined functions/types (all referenced symbols are either imported or defined in this task).

**3. Type consistency:**

- `_EXAMPLES: list[dict]` — module-level constant, type matches usage in `build_translation_prompt`.
- `_length_target_vi_chars(start_ms: int, end_ms: int) -> int` — signature matches all 3 test usages.
- `TimedSegment` model unchanged.
- `_FakeAnthropic` reused from `tests/test_translate_responses.py` (existing import path).

**4. TDD discipline:**

- Step 1: write 5 failing tests.
- Step 2: verify all 5 fail (RED).
- Steps 3-5: implement (constants, prompt builder, system prompt).
- Step 6: verify all 5 pass (GREEN).
- Step 7: full suite regression check.
- Step 8: commit only after green.

The `test_translate_pipeline_produces_valid_response` integration test exercises the full `translate_with_llm` flow with the new prompts — confirms no behavior regression.
