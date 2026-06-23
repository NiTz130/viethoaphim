from __future__ import annotations

import csv
import json
from pathlib import Path

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


def build_translation_prompt(segments: list[TimedSegment], context_bundle: dict) -> str:
    payload = {
        "instructions": [
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
) -> list[TranslationRow]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for translation")
    if not model:
        raise RuntimeError("LLM_MODEL is required for translation")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Vietnamese localization editor for Chinese comedy cartoons."},
            {"role": "user", "content": build_translation_prompt(segments, context_bundle)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return [TranslationRow.model_validate(item) for item in data["translations"]]


def export_review_csv(path: Path, segments: list[TimedSegment], translations: list[TranslationRow]) -> None:
    by_id = {row.segment_id: row for row in translations}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for segment in segments:
            row = by_id.get(
                segment.id,
                TranslationRow(
                    segment_id=segment.id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    speaker=segment.speaker,
                    text_cn=segment.text,
                    text_vi="",
                ),
            )
            writer.writerow(row.model_dump())


def import_review_csv(path: Path) -> list[TranslationRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [TranslationRow.model_validate(row) for row in reader]
