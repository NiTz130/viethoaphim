from __future__ import annotations

import csv
import json
import re
import sys
import time
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from .models import TimedSegment, TranslationRow


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


def parse_translation_response(content: str) -> list[TranslationRow]:
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

    _warn_oversized_translations(rows)

    return rows


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
) -> list[TranslationRow]:
    """Translate segments by batching them into LLM calls.

    A single LLM call cannot handle hundreds of segments within max_tokens
    limits, so we batch into groups of LLM_BATCH_SIZE and concatenate results.
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

    all_rows: list[TranslationRow] = []
    for start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[start:start + LLM_BATCH_SIZE]
        all_rows.extend(_translate_one_batch_with_retry(batch, context_bundle, settings))
    return all_rows


def _translate_one_batch(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
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
    user_prompt = build_translation_prompt(segments, context_bundle)

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
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors.

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
            return _translate_one_batch(segments, context_bundle, settings)
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
