from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MemoryWarningItem(BaseModel):
    path: str
    message: str


class TranslationExample(BaseModel):
    text_cn: str
    text_vi: str
    source_job: str
    source: str
    confidence: float
    segment_id: str = ""


class MemoryCharacter(BaseModel):
    name_cn: str
    name_vi: str
    source_job: str
    source: str
    confidence: float


class MemoryGlossaryEntry(BaseModel):
    source_text: str
    target: str
    source_job: str
    source: str
    confidence: float


class SystemMemory(BaseModel):
    translation_examples: list[TranslationExample] = Field(default_factory=list)
    characters: list[MemoryCharacter] = Field(default_factory=list)
    glossary: list[MemoryGlossaryEntry] = Field(default_factory=list)
    warnings: list[MemoryWarningItem] = Field(default_factory=list)


def collect_system_memory(jobs_dir: Path, current_job: Path | None = None) -> SystemMemory:
    warnings: list[MemoryWarningItem] = []
    examples: dict[str, TranslationExample] = {}
    characters: dict[str, MemoryCharacter] = {}
    glossary: dict[str, MemoryGlossaryEntry] = {}

    for job_dir in _iter_job_dirs(jobs_dir, current_job):
        _collect_review_csv(job_dir, examples, warnings)
        _collect_translated_json(job_dir, examples, warnings)
        _collect_characters_json(job_dir, characters, warnings)
        _collect_glossary_json(job_dir, glossary, warnings)
        _collect_reference_context_json(job_dir, characters, glossary, warnings)

    return SystemMemory(
        translation_examples=list(examples.values()),
        characters=list(characters.values()),
        glossary=list(glossary.values()),
        warnings=warnings,
    )


def _iter_job_dirs(jobs_dir: Path, current_job: Path | None) -> list[Path]:
    if not jobs_dir.exists():
        return []
    current = current_job.resolve() if current_job else None
    job_dirs: list[Path] = []
    for child in sorted(jobs_dir.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        if current is not None and child.resolve() == current:
            continue
        job_dirs.append(child)
    return job_dirs


def _collect_review_csv(
    job_dir: Path,
    examples: dict[str, TranslationExample],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "translation" / "review.csv"
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text_cn = (row.get("text_cn") or "").strip()
                text_vi = (row.get("text_vi") or "").strip()
                status = (row.get("status") or "draft").strip()
                if not text_cn or not text_vi or status == "skip":
                    continue
                confidence = 0.95 if status == "reviewed" else 0.70
                _put_example(
                    examples,
                    TranslationExample(
                        text_cn=text_cn,
                        text_vi=text_vi,
                        source_job=job_dir.name,
                        source="translation/review.csv",
                        confidence=confidence,
                        segment_id=(row.get("segment_id") or "").strip(),
                    ),
                )
    except OSError as exc:
        warnings.append(MemoryWarningItem(path=str(path), message=f"Could not read CSV: {exc}"))


def _collect_translated_json(
    job_dir: Path,
    examples: dict[str, TranslationExample],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "translation" / "translated.json"
    data = _load_json(path, warnings)
    if data is None:
        return
    rows = data.get("translations", []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        warnings.append(MemoryWarningItem(path=str(path), message="Expected a JSON list or object with translations list"))
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        text_cn = str(row.get("text_cn") or "").strip()
        text_vi = str(row.get("text_vi") or "").strip()
        status = str(row.get("status") or "draft").strip()
        if not text_cn or not text_vi or status == "skip":
            continue
        _put_example(
            examples,
            TranslationExample(
                text_cn=text_cn,
                text_vi=text_vi,
                source_job=job_dir.name,
                source="translation/translated.json",
                confidence=0.60,
                segment_id=str(row.get("segment_id") or "").strip(),
            ),
        )


def _collect_characters_json(
    job_dir: Path,
    characters: dict[str, MemoryCharacter],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "context" / "characters.json"
    data = _load_json(path, warnings)
    if data is None:
        return
    if not isinstance(data, list):
        warnings.append(MemoryWarningItem(path=str(path), message="Expected characters.json to contain a JSON list"))
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        name_cn = str(item.get("name_cn") or "").strip()
        name_vi = str(item.get("name_vi") or "").strip()
        if not name_cn or not name_vi:
            continue
        _put_character(
            characters,
            MemoryCharacter(
                name_cn=name_cn,
                name_vi=name_vi,
                source_job=job_dir.name,
                source="context/characters.json",
                confidence=float(item.get("confidence") or 0.80),
            ),
        )


def _collect_glossary_json(
    job_dir: Path,
    glossary: dict[str, MemoryGlossaryEntry],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "context" / "glossary.json"
    data = _load_json(path, warnings)
    if data is None:
        return
    if not isinstance(data, dict):
        warnings.append(MemoryWarningItem(path=str(path), message="Expected glossary.json to contain a JSON object"))
        return
    for source_text, raw_target in data.items():
        source = str(source_text).strip()
        target, confidence = _glossary_target_and_confidence(raw_target, 0.80)
        if not source or not target:
            continue
        _put_glossary(
            glossary,
            MemoryGlossaryEntry(
                source_text=source,
                target=target,
                source_job=job_dir.name,
                source="context/glossary.json",
                confidence=confidence,
            ),
        )


def _collect_reference_context_json(
    job_dir: Path,
    characters: dict[str, MemoryCharacter],
    glossary: dict[str, MemoryGlossaryEntry],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "context" / "reference_context.json"
    data = _load_json(path, warnings)
    if data is None:
        return
    if not isinstance(data, dict):
        warnings.append(MemoryWarningItem(path=str(path), message="Expected reference_context.json to contain a JSON object"))
        return
    for filename, entries in data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source_text = str(entry.get("source") or "").strip()
            target = str(entry.get("target") or "").strip()
            if not source_text or not target:
                continue
            if filename == "Names.txt":
                _put_character(
                    characters,
                    MemoryCharacter(
                        name_cn=source_text,
                        name_vi=target.split("/")[0],
                        source_job=job_dir.name,
                        source="context/reference_context.json",
                        confidence=0.65,
                    ),
                )
            else:
                _put_glossary(
                    glossary,
                    MemoryGlossaryEntry(
                        source_text=source_text,
                        target=target,
                        source_job=job_dir.name,
                        source="context/reference_context.json",
                        confidence=0.60,
                    ),
                )


def _load_json(path: Path, warnings: list[MemoryWarningItem]) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        warnings.append(MemoryWarningItem(path=str(path), message=f"Invalid JSON: {exc.msg}"))
    except OSError as exc:
        warnings.append(MemoryWarningItem(path=str(path), message=f"Could not read JSON: {exc}"))
    return None


def _glossary_target_and_confidence(raw_target: Any, default_confidence: float) -> tuple[str, float]:
    if isinstance(raw_target, dict):
        target = str(raw_target.get("target") or raw_target.get("name_vi") or "").strip()
        confidence = float(raw_target.get("confidence") or default_confidence)
        return target, confidence
    return str(raw_target or "").strip(), default_confidence


def _put_example(examples: dict[str, TranslationExample], candidate: TranslationExample) -> None:
    current = examples.get(candidate.text_cn)
    if current is None or candidate.confidence > current.confidence:
        examples[candidate.text_cn] = candidate


def _put_character(characters: dict[str, MemoryCharacter], candidate: MemoryCharacter) -> None:
    current = characters.get(candidate.name_cn)
    if current is None or candidate.confidence > current.confidence:
        characters[candidate.name_cn] = candidate


def _put_glossary(glossary: dict[str, MemoryGlossaryEntry], candidate: MemoryGlossaryEntry) -> None:
    current = glossary.get(candidate.source_text)
    if current is None or candidate.confidence > current.confidence:
        glossary[candidate.source_text] = candidate
