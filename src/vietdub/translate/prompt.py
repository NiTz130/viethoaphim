from __future__ import annotations

import json as _json
import logging
import re
from importlib.resources import files
from json import JSONDecodeError
from pathlib import Path as _Path

from pydantic import ValidationError

from ..models import TimedSegment, TranslationRow
from .csv import ALLOWED_REVIEW_STATUSES
from .length import _length_target_vi_chars, _warn_oversized_translations


_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")


def _load_few_shot_examples() -> list[dict]:
    """Load few-shot examples from the package data directory.

    Uses importlib.resources for installed packages, with a fallback
    to a relative path for running directly from the source tree.
    """
    data = files("vietdub").joinpath("data", "few_shot_examples.json")
    if data.is_file():
        return _json.loads(data.read_text(encoding="utf-8"))
    # Dev fallback: look for data/few_shot_examples.json relative to repo root.
    path = _Path(__file__).parent.parent.parent / "data" / "few_shot_examples.json"
    if path.is_file():
        return _json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"few_shot_examples.json not found in package data or at {path}"
    )


# Use the parent package's logger so test assertions on
# `r.name == "vietdub.translate"` continue to match.
logger = logging.getLogger("vietdub.translate")


def _strip_markdown_fences(content: str) -> str:
    """Strip leading/trailing markdown code fences (e.g. ```json\\n...\\n```).

    Handles standard (```json\\n...\\n```), single-line opening (```json {...}```),
    trailing-only (\\n```), and CRLF line endings.
    """
    stripped = _LEADING_FENCE_RE.sub("", content.strip(), count=1)
    stripped = _TRAILING_FENCE_RE.sub("", stripped, count=1)
    return stripped.strip()


def parse_translation_response(
    content: str,
    warn_oversized: bool = True,
) -> list[TranslationRow]:
    """Parse the LLM's JSON translation response into TranslationRow objects.

    Parameters
    ----------
    content:
        Raw response text from the LLM (may be wrapped in markdown fences).
    warn_oversized:
        If True (default), emit ``[LEN WARNING]`` lines to stderr for any
        translations that exceed ``target_vi_chars``. Set False when parsing
        the LLM's own review/refinement output, since warnings on the
        LLM's attempted fix would just be noise.

    Returns
    -------
    list[TranslationRow]
        Validated rows parsed from the JSON payload.
    """
    try:
        data = _json.loads(content)
    except JSONDecodeError as exc_raw:
        # Retry after stripping markdown code fences (LLMs sometimes wrap JSON in ```json ... ```)
        try:
            data = _json.loads(_strip_markdown_fences(content))
        except JSONDecodeError as exc_stripped:
            new_exc = RuntimeError(
                f"Invalid LLM translation response: invalid JSON at char {exc_stripped.pos}"
            )
            new_exc.add_note(
                f"Raw (pre-strip) JSON also failed at char {exc_raw.pos}: {exc_raw.msg}"
            )
            raise new_exc from exc_stripped

    if not isinstance(data, dict):
        raise RuntimeError("Invalid LLM translation response: expected a JSON object")

    translations = data.get("translations")
    if not isinstance(translations, list):
        raise RuntimeError("Invalid LLM translation response: expected 'translations' to be a list")

    rows: list[TranslationRow] = []
    for index, item in enumerate(translations, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid LLM translation response: translation item {index} is not an object")
        if item.get("status") not in ALLOWED_REVIEW_STATUSES:
            logger.warning(
                f"Warning: LLM returned invalid status {item.get('status')!r} "
                f"for segment {item.get('segment_id')!r}; coercing to 'draft'"
            )
            item = {**item, "status": "draft"}
        try:
            rows.append(TranslationRow.model_validate(item))
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid LLM translation response: translation item {index} "
                f"(segment_id={item.get('segment_id', '?')!r}) failed validation: {exc}"
            ) from exc

    if warn_oversized:
        _warn_oversized_translations(rows)

    return rows


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
            "OUTPUT FORMAT (CORRECT): {\"translations\": [{\"segment_id\": \"m-0001\", \"start_ms\": 0, \"end_ms\": 1000, \"speaker\": null, \"text_cn\": \"你好\", \"text_vi\": \"Xin chào\", \"context_note\": \"\", \"status\": \"draft\"}]}",
            "OUTPUT FORMAT (INCORRECT — do NOT do this):",
            "  [{\"segment_id\": \"m-0001\", ...}]  (bare array, missing top-level object)",
            "  Wrapped in markdown fences or extra braces",
        ],
        "examples": _load_few_shot_examples(),
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
    return _json.dumps(payload, ensure_ascii=False)