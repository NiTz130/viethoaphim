from __future__ import annotations

import csv
import json
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


def parse_translation_response(content: str) -> list[TranslationRow]:
    try:
        data = json.loads(content)
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
            item = {**item, "status": "draft"}
        try:
            rows.append(TranslationRow.model_validate(item))
        except ValidationError as exc:
            raise RuntimeError(f"Invalid LLM translation response: translation item {index} failed validation") from exc
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


def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    api_key: str,
    model: str,
    base_url: str = "",
) -> list[TranslationRow]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for translation")
    if not model:
        raise RuntimeError("LLM_MODEL is required for translation")

    from openai import APIStatusError, AuthenticationError, OpenAI, OpenAIError

    clean_base_url = base_url.rstrip("/")
    use_responses_endpoint = clean_base_url.endswith("/responses")
    client_kwargs = {"api_key": api_key}
    if clean_base_url:
        client_kwargs["base_url"] = clean_base_url.removesuffix("/responses") if use_responses_endpoint else clean_base_url
    client = OpenAI(**client_kwargs)
    try:
        prompt = build_translation_prompt(segments, context_bundle)
        if use_responses_endpoint:
            response = client.responses.create(
                model=model,
                instructions="You are a Vietnamese localization editor for Chinese comedy cartoons.",
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
            content = response.output_text or "{}"
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Vietnamese localization editor for Chinese comedy cartoons."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
    except AuthenticationError as exc:
        raise RuntimeError("OpenAI authentication failed; check OPENAI_API_KEY") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.status_code}") from exc
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.__class__.__name__}") from exc
    return parse_translation_response(content)


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
