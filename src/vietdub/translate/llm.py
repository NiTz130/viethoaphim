from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from ..models import TimedSegment, TranslationRow
from .glossary import (
    _cap_glossary,
    _load_ignore_list,
    _load_initial_glossary,
    _load_phrase_patterns,
    _load_pronouns,
    _update_glossary,
)
from .length import _length_target_vi_chars, _warn_oversized_translations
from .prompt import build_translation_prompt, parse_translation_response


LLM_BATCH_SIZE = 50
LLM_MAX_TOKENS = 8192
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_S = 1.0
LLM_RETRY_BACKOFF_FACTOR = 2.0

# Reference to the package-level LLM_BATCH_SIZE so test code can monkeypatch
# `vietdub.translate.LLM_BATCH_SIZE` and have the override take effect inside
# translate_with_llm. Updated lazily after __init__ re-export so monkeypatches
# on the package attribute propagate.
_PKG_LLM_BATCH_SIZE: int | None = None


def _resolve_batch_size() -> int:
    """Resolve current LLM_BATCH_SIZE, honoring test monkeypatches on the package."""
    import sys
    pkg = sys.modules.get("vietdub.translate")
    if pkg is not None:
        patched = getattr(pkg, "LLM_BATCH_SIZE", None)
        if isinstance(patched, int):
            return patched
    return LLM_BATCH_SIZE


def _resolve_max_tokens() -> int:
    """Resolve current LLM_MAX_TOKENS, honoring test monkeypatches on the package."""
    import sys
    pkg = sys.modules.get("vietdub.translate")
    if pkg is not None:
        patched = getattr(pkg, "LLM_MAX_TOKENS", None)
        if isinstance(patched, int):
            return patched
    return LLM_MAX_TOKENS


def _resolve_warned_unknown_caps() -> set[str]:
    """Resolve _warned_unknown_caps, honoring test monkeypatches on the package."""
    import sys
    pkg = sys.modules.get("vietdub.translate")
    if pkg is not None and isinstance(getattr(pkg, "_warned_unknown_caps", None), set):
        return pkg._warned_unknown_caps
    return _warned_unknown_caps


def _resolve_find_reference_data_dir():
    import sys
    pkg = sys.modules.get("vietdub.translate")
    if pkg is not None and hasattr(pkg, "find_reference_data_dir"):
        return pkg.find_reference_data_dir
    from ..reference import find_reference_data_dir as _frd
    return _frd

KNOWN_MODEL_OUTPUT_CAPS: dict[str, int] = {
    "MiniMax-M3": 8192,
    "MiniMax-M2": 8192,
    "claude-haiku-4-5-20251001": 8192,
    "claude-sonnet-4-6": 8192,
    "claude-opus-4-8": 8192,
    "claude-3-5-haiku-20241022": 8192,
}


_warned_unknown_caps: set[str] = set()


# Use the parent package's logger so test assertions on
# `r.name == "vietdub.translate"` continue to match.
logger = logging.getLogger("vietdub.translate")


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
    """Translate segments by batching them into LLM calls.

    A single LLM call cannot handle hundreds of segments within max_tokens
    limits, so we batch into groups of LLM_BATCH_SIZE and concatenate results.
    When review=True (default), each batch is followed by an LLM review pass
    that refines translations exceeding target_vi_chars.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for translation")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is required for translation")

    cap = KNOWN_MODEL_OUTPUT_CAPS.get(settings.llm_model)
    max_tokens = _resolve_max_tokens()
    if cap is not None and max_tokens > cap:
        raise RuntimeError(
            f"LLM_MAX_TOKENS ({max_tokens}) exceeds known output cap "
            f"({cap}) for model {settings.llm_model!r}. "
            f"Reduce LLM_MAX_TOKENS or use a different model."
        )
    warned = _resolve_warned_unknown_caps()
    if cap is None and settings.llm_model not in warned:
        warned.add(settings.llm_model)
        logger.warning(
            f"Warning: model {settings.llm_model!r} not in KNOWN_MODEL_OUTPUT_CAPS; "
            f"skipping output cap validation (warning shown once per model). "
            f"If you hit truncation, add the model's output cap to the dict."
        )

    glossary: dict[str, str] = dict(_load_initial_glossary())
    _cap_glossary(glossary)

    # Load 3 reference data sections (loaded once, reused across batches)
    try:
        ref_root = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        ref_dir = _resolve_find_reference_data_dir()(ref_root)
        pronoun_guide = _load_pronouns(ref_dir)
        phrase_patterns = _load_phrase_patterns(ref_dir)
        ignore_list = _load_ignore_list(ref_dir)
    except Exception as exc:
        logger.warning(
            f"Warning: failed to load reference data: {type(exc).__name__}: {exc}"
        )
        pronoun_guide = []
        phrase_patterns = []
        ignore_list = []

    all_rows: list[TranslationRow] = []
    batch_size = _resolve_batch_size()
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
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


def _review_batch_for_length(
    rows: list[TranslationRow],
    context_bundle: dict,
    settings,
) -> list[TranslationRow]:
    """Review translations for length compliance and return refined rows.

    Sends each row's text_cn, text_vi, and target_vi_chars to the LLM with
    instruction to refine translations that significantly exceed target.
    Returns a new list of rows with updated text_vi (matching by segment_id).
    """
    anthropic_mod = sys.modules.get("anthropic")
    if anthropic_mod is None:
        import anthropic as anthropic_mod
    api_connection_error = getattr(anthropic_mod, "APIConnectionError", ())
    api_status_error = getattr(anthropic_mod, "APIStatusError", ())
    authentication_error = getattr(anthropic_mod, "AuthenticationError", ())

    review_payload = {
        "instructions": [
            "Review these Vietnamese translations for length compliance.",
            "For each segment, target_vi_chars is the budget based on segment duration "
            "(Vietnamese ≈ 14 chars/sec + 10% buffer).",
            "Refine any translation that significantly exceeds its target_vi_chars "
            "(make it shorter while preserving meaning and tone).",
            "For translations already within budget or under budget, return unchanged.",
            "Preserve segment_id and original meaning; only adjust text_vi for length.",
            "Return ALL fields for each segment: segment_id, start_ms, end_ms, speaker, "
            "text_cn, text_vi, context_note, status. (Required: start_ms and end_ms must be present.)",
            "Return JSON only with a top-level \"translations\" array.",
        ],
        "segments": [
            {
                "segment_id": row.segment_id,
                "start_ms": row.start_ms,
                "end_ms": row.end_ms,
                "speaker": row.speaker,
                "text_cn": row.text_cn,
                "text_vi": row.text_vi,
                "context_note": row.context_note,
                "status": row.status,
                "target_vi_chars": _length_target_vi_chars(row.start_ms, row.end_ms),
            }
            for row in rows
        ],
    }

    client = anthropic_mod.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=_resolve_max_tokens(),
            system=(
                "You are a Vietnamese translation editor refining for length. "
                "Return JSON only with a top-level \"translations\" array."
            ),
            messages=[{"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
        )
    except api_connection_error:
        raise
    except api_status_error:
        raise
    except authentication_error:
        raise

    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax review returned empty response (no text content blocks)")
    content = "".join(text_parts)
    refined_rows = parse_translation_response(content, warn_oversized=False)

    refined_map: dict[str, str] = {r.segment_id: r.text_vi for r in refined_rows}

    output = []
    for row in rows:
        new_text = refined_map.get(row.segment_id, "").strip()
        if new_text:
            output.append(row.model_copy(update={"text_vi": new_text}))
        else:
            output.append(row)
    return output


def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
    glossary: dict[str, str] | None = None,
    pronoun_guide: list[dict[str, str]] | None = None,
    phrase_patterns: list[dict[str, str]] | None = None,
    ignore_list: list[str] | None = None,
) -> list[TranslationRow]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
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
    user_prompt = build_translation_prompt(
        segments, context_bundle, glossary=glossary,
        pronoun_guide=pronoun_guide,
        phrase_patterns=phrase_patterns,
        ignore_list=ignore_list,
    )

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=_resolve_max_tokens(),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
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
        raise RuntimeError("MiniMax returned empty response (no text content blocks)")
    content = "".join(text_parts)
    rows = parse_translation_response(content)

    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"MiniMax hit max_tokens ({_resolve_max_tokens()}) for batch of {len(segments)} segments "
            f"(translated {len(rows)}). Reduce LLM_BATCH_SIZE or increase max_tokens."
        )
    if len(rows) != len(segments):
        raise RuntimeError(
            f"MiniMax returned {len(rows)} rows for batch of {len(segments)} segments; "
            f"count mismatch (likely truncation). Reduce LLM_BATCH_SIZE."
        )

    return rows


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
    """Call _translate_one_batch with exponential-backoff retry on transient errors.

    If review=True, runs a second LLM call per batch to refine translations
    for length compliance.

    Catches bare `anthropic.*` exceptions (raised unwrapped from
    `_translate_one_batch`) and retries on transient failures (network errors,
    HTTP 429, 5xx). Non-retryable API errors are wrapped with user-facing
    messages. Truncation and other structural `RuntimeError`s raised by
    `_translate_one_batch` propagate unchanged — they are not caught here.
    The `anthropic` module is resolved via `sys.modules` so test mocks that
    omit exception classes degrade gracefully to a pass-through.
    """
    anthropic_mod = sys.modules.get("anthropic")
    if anthropic_mod is None:
        import anthropic as anthropic_mod
    api_connection_error = getattr(anthropic_mod, "APIConnectionError", ())
    api_status_error = getattr(anthropic_mod, "APIStatusError", ())
    authentication_error = getattr(anthropic_mod, "AuthenticationError", ())

    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            rows = _translate_one_batch(
                segments, context_bundle, settings, glossary=glossary,
                pronoun_guide=pronoun_guide,
                phrase_patterns=phrase_patterns,
                ignore_list=ignore_list,
            )
            if review:
                rows = _review_batch_for_length(rows, context_bundle, settings)
            return rows
        except api_connection_error as exc:
            last_exc = exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
        except api_status_error as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in {408, 425, 429, 500, 502, 503, 504, 524}:
                if isinstance(exc, authentication_error):
                    raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
                raise RuntimeError(f"MiniMax HTTP {status}: {exc.message}") from exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))

    raise RuntimeError(
        f"MiniMax batch failed after {LLM_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc