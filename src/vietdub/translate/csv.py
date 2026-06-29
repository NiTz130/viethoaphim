from __future__ import annotations

import csv
from pathlib import Path

from pydantic import ValidationError

from ..models import TimedSegment, TranslationRow


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