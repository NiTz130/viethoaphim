from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from .models import TimedSegment, TranslationRow
from .reference import find_reference_data_dir, load_dictionary_entries


CSV_FIELDS = [
    "segment_id",
    "start_ms",
    "end_ms",
    "speaker",
    "text_cn",
    "text_vi",
    "context_note",
    "status",
]

ALLOWED_REVIEW_STATUSES = {"draft", "reviewed", "skip"}


_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")

_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")
_MAX_GLOSSARY_ENTRIES = 50


_EXAMPLES: list[dict] = [
    {
        "segment_id": "ex-1",
        "start_ms": 0,
        "end_ms": 2440,
        "speaker": "",
        "text_cn": "你这个笨蛋！",
        "text_vi": "Mày ngu vậy!",
        "context_note": "",
        "status": "draft",
    },
    {
        "segment_id": "ex-2",
        "start_ms": 2440,
        "end_ms": 4320,
        "speaker": "",
        "text_cn": "长老，我们该怎么办？",
        "text_vi": "Trưởng lão, giờ chúng ta phải làm sao?",
        "context_note": "",
        "status": "draft",
    },
    {
        "segment_id": "ex-3",
        "start_ms": 4320,
        "end_ms": 5960,
        "speaker": "",
        "text_cn": "哈哈哈哈，你真是太有趣了！",
        "text_vi": "Hahaha, mày buồn cười thiệt chứ!",
        "context_note": "",
        "status": "draft",
    },
]


def _length_target_vi_chars(start_ms: int, end_ms: int) -> int:
    """Compute target Vietnamese character count for a segment.

    Vietnamese conversational speech is ~14 chars/sec; add 10% buffer for
    punctuation and natural variation. Floor at 10 chars for very short segments.
    """
    duration_s = (end_ms - start_ms) / 1000.0
    return max(10, int(duration_s * 14 * 1.1))


_LENGTH_TOLERANCE = 0.20  # ±20% of target_vi_chars


def _warn_oversized_translations(rows: list[TranslationRow]) -> None:
    """Warn if Vietnamese translation length is outside ±20% of target.

    Skips empty translations and segments in the floor region (target <= 10).
    Warnings go to stderr with [LEN WARNING] prefix for easy filtering.
    """
    for row in rows:
        text_len = len(row.text_vi)
        if text_len == 0:
            continue
        target = _length_target_vi_chars(row.start_ms, row.end_ms)
        if target <= 10:
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
        "examples": _EXAMPLES,
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
    return json.dumps(payload, ensure_ascii=False)


LLM_BATCH_SIZE = 50
LLM_MAX_TOKENS = 8192
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_S = 1.0
LLM_RETRY_BACKOFF_FACTOR = 2.0

KNOWN_MODEL_OUTPUT_CAPS: dict[str, int] = {
    "MiniMax-M3": 8192,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-sonnet-20240620": 8192,
    "claude-3-opus-20240229": 4096,
}


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
    if cap is not None and LLM_MAX_TOKENS > cap:
        raise RuntimeError(
            f"LLM_MAX_TOKENS ({LLM_MAX_TOKENS}) exceeds known output cap "
            f"({cap}) for model {settings.llm_model!r}. "
            f"Reduce LLM_MAX_TOKENS or use a different model."
        )
    if cap is None:
        print(
            f"Warning: model {settings.llm_model!r} not in KNOWN_MODEL_OUTPUT_CAPS; "
            f"skipping output cap validation. If you hit truncation, "
            f"add the model's output cap to the dict.",
            file=sys.stderr,
        )

    glossary: dict[str, str] = dict(_load_initial_glossary())
    _cap_glossary(glossary)

    # Load 3 reference data sections (loaded once, reused across batches)
    try:
        ref_root = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        ref_dir = find_reference_data_dir(ref_root)
        pronoun_guide = _load_pronouns(ref_dir)
        phrase_patterns = _load_phrase_patterns(ref_dir)
        ignore_list = _load_ignore_list(ref_dir)
    except Exception as exc:
        print(
            f"Warning: failed to load reference data: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
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


def _load_initial_glossary() -> dict[str, str]:
    """Load initial name glossary from reference data (Names.txt).

    Uses the shared ``load_dictionary_entries`` helper, which parses the
    real ``cn_name=vi_name`` format. The reference directory is located via
    ``find_reference_data_dir`` so it works with both flat and nested
    layouts (e.g. ``data/Data của thtgiang (đọc README)/``). Returns an
    empty dict if the file is missing or unreadable.
    """
    glossary: dict[str, str] = {}
    try:
        ref_root = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        ref_dir = find_reference_data_dir(ref_root)
        glossary = load_dictionary_entries(ref_dir / "Names.txt")
    except Exception as exc:
        print(
            f"Warning: failed to load initial glossary: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        glossary = {}
    return glossary


def _load_pronouns(ref_dir: Path) -> list[dict[str, str]]:
    """Load pronoun guide from Pronouns.txt. Format: cn=vi per line.

    Uses the shared load_dictionary_entries helper (handles = separator).
    Returns empty list if file missing.
    """
    entries = load_dictionary_entries(ref_dir / "Pronouns.txt")
    return [{"source": cn, "target": vi} for cn, vi in entries.items()]


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


def _update_glossary(
    glossary: dict[str, str],
    rows: list[TranslationRow],
    batch: list[TimedSegment],
) -> None:
    """Extract name_cn → name_vi pairs from completed batch and merge into glossary.

    Heuristic: for each segment, find Chinese name runs (2-4 CJK chars) and
    align them positionally with capitalized Vietnamese words.
    """
    for row, seg in zip(rows, batch):
        cn_names = _CHINESE_NAME_RE.findall(seg.text)
        if not cn_names:
            continue
        vi_words = row.text_vi.split()
        for i, cn_name in enumerate(cn_names):
            if i >= len(vi_words):
                break
            vi_word = vi_words[i].strip(".,!?;:")
            if len(vi_word) >= 2 and vi_word[0].isupper():
                glossary.setdefault(cn_name, vi_word)


def _cap_glossary(
    glossary: dict[str, str],
    max_entries: int = _MAX_GLOSSARY_ENTRIES,
) -> None:
    """Cap glossary to max_entries by dropping oldest entries (FIFO)."""
    while len(glossary) > max_entries:
        oldest_key = next(iter(glossary))
        del glossary[oldest_key]


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
            max_tokens=LLM_MAX_TOKENS,
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
            max_tokens=LLM_MAX_TOKENS,
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
            f"MiniMax hit max_tokens ({LLM_MAX_TOKENS}) for batch of {len(segments)} segments "
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
            if status not in {429, 500, 502, 503, 504}:
                if isinstance(exc, authentication_error):
                    raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
                raise RuntimeError(f"MiniMax HTTP {status}: {exc.message}") from exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))

    raise RuntimeError(
        f"MiniMax batch failed after {LLM_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc


def export_review_csv(path: Path, segments: list[TimedSegment], translations: list[TranslationRow]) -> None:
    by_id = {row.segment_id: row for row in translations}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for segment in segments:
            translated = by_id.get(segment.id)
            row = TranslationRow(
                segment_id=segment.id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                speaker=segment.speaker,
                text_cn=segment.text,
                text_vi=translated.text_vi if translated else "",
                context_note=translated.context_note if translated else "",
                status=translated.status if translated else "draft",
            )
            writer.writerow(row.model_dump())


def import_review_csv(path: Path) -> list[TranslationRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in CSV_FIELDS if field not in fieldnames]
        if missing:
            raise RuntimeError(f"Invalid review CSV {path}: missing columns: {', '.join(missing)}")

        rows: list[TranslationRow] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                translation = TranslationRow.model_validate(row)
                if translation.end_ms <= translation.start_ms:
                    raise ValueError("end_ms must be greater than start_ms")
                rows.append(translation)
            except (ValidationError, ValueError) as exc:
                raise RuntimeError(f"Invalid review CSV {path}: row {row_number} failed validation") from exc
        return rows
