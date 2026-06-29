from __future__ import annotations

import csv
import json
import math
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
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
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
    data, warning = _load_json(path)
    if warning is not None:
        warnings.append(warning)
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
    data, warning = _load_json(path)
    if warning is not None:
        warnings.append(warning)
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
        confidence = _parse_confidence(item.get("confidence"), 0.80, path, warnings, name_cn)
        if confidence is None:
            continue
        _put_character(
            characters,
            MemoryCharacter(
                name_cn=name_cn,
                name_vi=name_vi,
                source_job=job_dir.name,
                source="context/characters.json",
                confidence=confidence,
            ),
        )


def _collect_glossary_json(
    job_dir: Path,
    glossary: dict[str, MemoryGlossaryEntry],
    warnings: list[MemoryWarningItem],
) -> None:
    path = job_dir / "context" / "glossary.json"
    data, warning = _load_json(path)
    if warning is not None:
        warnings.append(warning)
    if data is None:
        return
    if not isinstance(data, dict):
        warnings.append(MemoryWarningItem(path=str(path), message="Expected glossary.json to contain a JSON object"))
        return
    for source_text, raw_target in data.items():
        source = str(source_text).strip()
        target, confidence = _glossary_target_and_confidence(raw_target, 0.80, path, warnings, source)
        if not source or not target or confidence is None:
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
    data, warning = _load_json(path)
    if warning is not None:
        warnings.append(warning)
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


def _load_json(path: Path) -> tuple[Any, "MemoryWarningItem | None"]:
    """Load JSON from path, distinguishing missing-file from corruption.

    Returns:
        (data, None) on success.
        (None, None) if the file does not exist (caller treats as no-data).
        (None, warning) if the file exists but cannot be parsed or read.
    """
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace")), None
    except json.JSONDecodeError as exc:
        return None, MemoryWarningItem(path=str(path), message=f"Invalid JSON: {exc.msg}")
    except OSError as exc:
        return None, MemoryWarningItem(path=str(path), message=f"Could not read JSON: {exc}")


def _parse_confidence(
    raw_value: Any,
    default_confidence: float,
    path: Path,
    warnings: list[MemoryWarningItem],
    label: str,
) -> float | None:
    if raw_value is None or raw_value == "":
        return default_confidence
    try:
        confidence = float(raw_value)
    except (TypeError, ValueError):
        warnings.append(MemoryWarningItem(path=str(path), message=f"Invalid confidence for {label}: {raw_value!r}"))
        return None
    if not math.isfinite(confidence):
        warnings.append(MemoryWarningItem(path=str(path), message=f"Invalid confidence for {label}: {raw_value!r}"))
        return None
    return confidence


def _glossary_target_and_confidence(
    raw_target: Any,
    default_confidence: float,
    path: Path,
    warnings: list[MemoryWarningItem],
    label: str,
) -> tuple[str, float | None]:
    if isinstance(raw_target, dict):
        target = str(raw_target.get("target") or raw_target.get("name_vi") or "").strip()
        confidence = _parse_confidence(raw_target.get("confidence"), default_confidence, path, warnings, label)
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


def select_relevant_memory(
    memory: SystemMemory,
    segments: list,
    max_examples: int = 40,
    max_characters: int = 80,
    max_glossary: int = 120,
) -> dict[str, list[dict[str, Any]]]:
    transcript = "\n".join(str(segment.text) for segment in segments)
    examples = _rank_translation_examples(memory.translation_examples, transcript)[:max_examples]
    characters = _rank_memory_items(memory.characters, transcript, "name_cn")[:max_characters]
    glossary = _rank_memory_items(memory.glossary, transcript, "source_text")[:max_glossary]
    return {
        "translation_examples": [item.model_dump() for item in examples],
        "characters": [item.model_dump() for item in characters],
        "glossary": [item.model_dump() for item in glossary],
    }


def _rank_translation_examples(examples: list[TranslationExample], transcript: str) -> list[TranslationExample]:
    scored: list[tuple[float, float, TranslationExample]] = []
    for example in examples:
        score = _relevance_score(example.text_cn, transcript)
        if score <= 0:
            continue
        scored.append((example.confidence, score, example))
    scored.sort(key=lambda item: (item[0], item[1], len(item[2].text_cn)), reverse=True)
    return [item[2] for item in scored]


def _rank_memory_items(items: list[Any], transcript: str, source_attr: str) -> list[Any]:
    scored: list[tuple[float, float, int, Any]] = []
    for item in items:
        source_text = str(getattr(item, source_attr))
        score = _relevance_score(source_text, transcript)
        if score <= 0:
            continue
        scored.append((float(item.confidence), score, len(source_text), item))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in scored]


def _relevance_score(source_text: str, transcript: str) -> float:
    if not source_text or not transcript:
        return 0.0
    if source_text in transcript:
        return 1.0
    source_chars = {char for char in source_text if not char.isspace()}
    if not source_chars:
        return 0.0
    transcript_chars = {char for char in transcript if not char.isspace()}
    overlap = len(source_chars & transcript_chars) / len(source_chars)
    return overlap if overlap >= 0.50 else 0.0
