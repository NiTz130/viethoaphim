# Translation Multi-Pass Review — Design

**Date:** 2026-06-27
**Scope:** `src/vietdub/translate.py` (new `_review_batch_for_length` function + wiring) + `tests/test_translate.py`
**Status:** Approved (design), pending implementation

## Context

Specs 1-3 added:
- **Spec 1**: Better prompt with examples, tone, length targets (`target_vi_chars`)
- **Spec 2**: Post-processing length check that warns (stderr `[LEN WARNING]`) when translations exceed ±20% of target

Spec 2 produces noisy stderr on real jobs because the 14 chars/sec target is aspirational — Vietnamese translations naturally shorter or longer than the target produce many warnings. Operators ignore them.

This is Spec 4 of 4 (Prompt → Length → Glossary → Multi-pass Review). It's the **last spec** in the series and addresses the noise from Spec 2 directly by introducing a length-focused refinement pass.

## Goal

Add a per-batch review pass that refines translations to better match `target_vi_chars`. After the initial translation, a second LLM call reviews each translation and returns a refined version if the length is significantly off. Refined translations replace originals.

The feature is **opt-in** via a `review: bool` parameter on `translate_with_llm` (default `True` to fix Spec 2's noise by default; can be set `False` to skip and save LLM cost).

## Non-goals

- Auto-retry loop (Spec 4 is single-pass, not iterative).
- Refining for tone, names consistency, or other criteria — length only (per user choice).
- Skipping the review pass when length is already within budget — saves one LLM call but requires extra pre-check; cost vs benefit unclear.
- Operator-configurable review criteria — fixed prompt for now.
- Cross-batch consistency checks via review pass — covered by Spec 3 (glossary).

## Design

### Section 1: `_review_batch_for_length` function

Add a new private function in `src/vietdub/translate.py`:

```python
def _review_batch_for_length(
    rows: list[TranslationRow],
    context_bundle: dict,
    settings,
) -> list[TranslationRow]:
    """Review translations for length compliance and return refined rows.
    
    Sends each row's text_cn, text_vi, and target_vi_chars to the LLM with
    instruction to refine translations that significantly exceed target.
    Returns a new list of rows with updated text_vi (matching by segment_id).
    
    Cost: 1 additional LLM call per batch.
    """
    import anthropic

    review_payload = {
        "instructions": [
            "Review these Vietnamese translations for length compliance.",
            "For each segment, target_vi_chars is the budget based on segment duration "
            "(Vietnamese ≈ 14 chars/sec + 10% buffer).",
            "Refine any translation that significantly exceeds its target_vi_chars "
            "(make it shorter while preserving meaning and tone).",
            "For translations already within budget or under budget, return unchanged.",
            "Preserve segment_id and original meaning; only adjust text_vi for length.",
            "Return JSON only with a top-level \"translations\" array.",
        ],
        "segments": [
            {
                "segment_id": row.segment_id,
                "text_cn": row.text_cn,
                "text_vi": row.text_vi,
                "target_vi_chars": _length_target_vi_chars(row.start_ms, row.end_ms),
            }
            for row in rows
        ],
    }

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=LLM_MAX_TOKENS,
            system="You are a Vietnamese translation editor refining for length. Return JSON only with a top-level \"translations\" array.",
            messages=[{"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
        )
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIStatusError:
        raise
    except anthropic.APIConnectionError:
        raise

    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax review returned empty response (no text content blocks)")
    content = "".join(text_parts)
    refined_rows = parse_translation_response(content)

    # Build map for replacement
    refined_map: dict[str, str] = {r.segment_id: r.text_vi for r in refined_rows}

    # Apply refined text_vi to originals (preserve all other fields)
    output = []
    for row in rows:
        new_text = refined_map.get(row.segment_id, "").strip()
        if new_text:
            output.append(row.model_copy(update={"text_vi": new_text}))
        else:
            output.append(row)
    return output
```

### Section 2: Wire into `_translate_one_batch_with_retry` and `translate_with_llm`

Add `review: bool = True` parameter to `_translate_one_batch_with_retry`:

```python
def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    review: bool = True,
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors.
    
    If review=True, runs a second LLM call per batch to refine translations
    for length compliance.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            rows = _translate_one_batch(segments, context_bundle, settings, glossary=glossary)
            if review:
                rows = _review_batch_for_length(rows, context_bundle, settings)
            return rows
        except (...) as exc:
            # ... existing retry logic ...
```

Add same `review: bool = True` parameter to `translate_with_llm` and pass through:

```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
    review: bool = True,
) -> list[TranslationRow]:
    """..."""
    # ... existing validation ...
    # ... existing batching loop ...
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        rows = _translate_one_batch_with_retry(
            batch, context_bundle, settings, glossary=glossary, review=review
        )
        all_rows.extend(rows)
        _update_glossary(glossary, rows, batch)
        _cap_glossary(glossary)
    return all_rows
```

### Section 3: Update CLI to expose `review` flag

The CLI (`vietdub run`) should accept `--review/--no-review` to control the flag:

```python
@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", ...),
    series: str | None = typer.Option(None, "--series", ...),
    no_review: bool = typer.Option(False, "--no-review", help="Skip length review pass (saves LLM cost)."),
) -> None:
    ...
    settings = Settings()
    job = run_review_pipeline(
        video=video,
        jobs_dir=Path(settings.jobs_dir),
        series=series,
        settings=settings,
        review=not no_review,  # inverted flag
    )
```

## Test plan

Add 5 tests to `tests/test_translate.py`:

1. **`test_review_batch_for_length_returns_refined_translations`**
   ```python
   def test_review_batch_for_length_returns_refined_translations(monkeypatch):
       """Review pass returns refined translations that match length target."""
       # Mock _review_batch_for_length to return rows with shorter text_vi
       # Verify final rows have the refined text_vi
   ```

2. **`test_review_batch_for_length_preserves_unchanged_translations`**
   ```python
   def test_review_batch_for_length_preserves_unchanged_translations(monkeypatch):
       """Review pass preserves translations already within budget."""
       # Mock to return same text_vi
       # Verify output equals input
   ```

3. **`test_review_batch_for_length_handles_partial_review_response`**
   ```python
   def test_review_batch_for_length_handles_partial_review_response(monkeypatch):
       """If review returns only some segments, keep originals for missing ones."""
       # Mock to return 1 of 3 rows
       # Verify 1 row updated, 2 rows unchanged
   ```

4. **`test_review_batch_for_length_skips_empty_refinement`**
   ```python
   def test_review_batch_for_length_skips_empty_refinement(monkeypatch):
       """If review returns empty text_vi for a row, keep original."""
       # Mock to return text_vi="" for some row
       # Verify original text_vi kept
   ```

5. **`test_translate_with_llm_runs_review_pass_when_enabled`**
   ```python
   def test_translate_with_llm_runs_review_pass_when_enabled(monkeypatch):
       """translate_with_llm calls LLM twice per batch when review=True."""
       # Mock anthropic; count calls
       # Run with review=True → expect 2 calls per batch
       # Run with review=False → expect 1 call per batch
   ```

## Risk

**Medium.** Doubles LLM cost per job. Mitigation: opt-out via `--no-review` flag for cost-sensitive operators. The refinement should also reduce Spec 2's stderr noise, making logs more useful.

## Rollback

Single source file change + tests. Revert the implementation commit if cost is unacceptable. The feature is opt-in (default `review=True` for best quality, but easy to flip to `review=False` if cost concerns emerge).

## Implementation notes

- `_review_batch_for_length` reuses `parse_translation_response` for output parsing — same error handling, fence stripping, validation as the main translation.
- The `model_copy(update={"text_vi": ...})` pattern preserves `start_ms`, `end_ms`, `speaker`, `text_cn`, `context_note`, `status` from the original row.
- LLM calls inside the review function are NOT retried separately — they go through the outer `_translate_one_batch_with_retry` loop (which retries the entire translation + review cycle).
- The review prompt doesn't include `examples` or `consistency_terms` (glossary) — the LLM only needs the existing translations + length targets to do its job.
- Existing 140 tests continue to pass — they use `_FakeAnthropic` which doesn't differentiate translate vs review calls (all mocked the same way). May need adjustment for tests that count LLM calls.
- Cost impact: 2x LLM calls per batch. For a 200-segment job (4 batches), this is +4 LLM calls. Significant but bounded.
