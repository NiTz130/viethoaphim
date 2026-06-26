from __future__ import annotations

import csv
import json
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


def _strip_markdown_fences(content: str) -> str:
    """Strip leading/trailing markdown code fences (e.g. ```json\\n...\\n```)."""
    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def parse_translation_response(content: str) -> list[TranslationRow]:
    try:
        data = json.loads(content)
    except JSONDecodeError:
        # Retry after stripping markdown code fences (LLMs sometimes wrap JSON in ```json ... ```)
        try:
            data = json.loads(_strip_markdown_fences(content))
        except JSONDecodeError as exc:
            raise RuntimeError(f"Invalid LLM translation response: invalid JSON at char {exc.pos}") from exc

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
    return rows


def build_translation_prompt(segments: list[TimedSegment], context_bundle: dict) -> str:
    payload = {
        "instructions": [
            "Return JSON only with a top-level \"translations\" array.",
            "Each translation must include segment_id, start_ms, end_ms, speaker, text_cn, text_vi, context_note, and status.",
            "Translate Chinese cartoon dialogue into natural Vietnamese.",
            "Use a silly, meme-friendly tone when the source is comedic.",
            "Keep Vietnamese lines short enough for dubbing timing.",
            "Preserve names and pronouns using the supplied context.",
        ],
        "context": context_bundle,
        "segments": [segment.model_dump() for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False)


LLM_BATCH_SIZE = 50
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_S = 1.0
LLM_RETRY_BACKOFF_FACTOR = 2.0


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
        "You are a Vietnamese localization editor for Chinese comedy cartoons. "
        "Return JSON only with a top-level \"translations\" array."
    )
    user_prompt = build_translation_prompt(segments, context_bundle)

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"MiniMax HTTP {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(f"MiniMax network error: {exc}") from exc

    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax returned empty response (no text content blocks)")
    content = "".join(text_parts)
    return parse_translation_response(content)


def _translate_one_batch_with_retry(
    segments: list[TimedSegment],
    context_bundle: dict,
    settings,
) -> list[TranslationRow]:
    """Call _translate_one_batch with exponential-backoff retry on transient errors.

    This helper handles both exception shapes from `_translate_one_batch`:
    the current shape (API errors wrapped as `RuntimeError` with the original
    `anthropic.*` exception preserved as `__cause__`) and the future shape
    (bare `anthropic.*` exceptions raised directly). Retries on transient
    errors (connection failures and 429/5xx status codes) with exponential
    backoff; wraps non-retryable failures and final exhausted retries as
    `RuntimeError`. The `anthropic` module is resolved via `sys.modules` so
    test mocks that omit exception classes degrade gracefully to a pass-through.
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
        except RuntimeError as exc:
            # Current path: _translate_one_batch wraps API errors as RuntimeError.
            # Unwrap the cause and apply the same retry logic as if the underlying
            # anthropic.* exception had been raised directly. Preserve the
            # inner RuntimeError message (do not re-wrap).
            cause = exc.__cause__ if exc.__cause__ is not None else exc
            if authentication_error and isinstance(cause, authentication_error):
                raise
            if api_connection_error and isinstance(cause, api_connection_error):
                last_exc = cause
                if attempt == LLM_MAX_RETRIES:
                    break
                time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
                continue
            if api_status_error and isinstance(cause, api_status_error):
                last_exc = cause
                status = getattr(cause, "status_code", None)
                if status not in {429, 500, 502, 503, 504}:
                    raise
                if attempt == LLM_MAX_RETRIES:
                    break
                time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
                continue
            # Non-API RuntimeError (e.g. empty response, parse error): don't retry.
            raise
        except api_connection_error as exc:
            # Future path: _translate_one_batch raises bare anthropic.APIConnectionError.
            last_exc = exc
            if attempt == LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_S * (LLM_RETRY_BACKOFF_FACTOR ** attempt))
        except api_status_error as exc:
            # Future path: _translate_one_batch raises bare anthropic.APIStatusError
            # (or its AuthenticationError subclass).
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in {429, 500, 502, 503, 504}:
                if authentication_error and isinstance(exc, authentication_error):
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
