# System Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a system-memory scanner that reuses prior `jobs/*` translation/context history to improve future Vietnamese translations and produce clear `context` artifacts for each new job.

**Architecture:** Add a focused `vietdub.memory` module that reads historical job artifacts, deduplicates them by confidence, and selects relevant memory for the current transcript. Wire the selected memory into `build_context_bundle()` and `run_review_pipeline()` so each job writes `system_memory.json`, `translation_examples.json`, merged `characters.json`, merged `glossary.json`, and scanner warnings when old files are malformed.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing Typer CLI, existing CSV/JSON job artifacts.

---

## File Structure

- Create `src/vietdub/memory.py`: collect historical memory from `jobs/*`, dedupe by confidence, filter memory to the current transcript, and return JSON-serializable dictionaries.
- Create `tests/test_memory.py`: unit tests for scanning `review.csv`, `translated.json`, `context/*.json`, prioritization, relevance filtering, current-job exclusion, and warning capture.
- Modify `src/vietdub/context.py`: accept selected system memory, merge it with reference-data characters/glossary, and expose `translation_examples` to the LLM prompt.
- Modify `tests/test_context.py`: cover system-memory fields and keep existing reference-context behavior.
- Modify `src/vietdub/pipeline.py`: call the scanner after merge/reference context and before translation, then write all context artifacts.
- Modify `tests/test_pipeline.py`: cover pipeline writing `system_memory.json`, `translation_examples.json`, and `system_memory_warnings.json`.
- Modify `README.md`: document that each review run now learns from previous jobs and where the memory artifacts are written.
- Modify `docs/manual-test.md`: add manual checks for the new `context` artifacts.

## Task 1: Add Historical Memory Scanner

**Files:**
- Create: `src/vietdub/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write failing scanner tests**

Create `tests/test_memory.py` with this content:

```python
import json

from vietdub.memory import collect_system_memory


def test_collect_system_memory_reads_all_supported_sources(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "old-job"
    (job / "translation").mkdir(parents=True)
    (job / "context").mkdir()
    (job / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin chao da sua,,reviewed\n"
        "m-0002,1000,2000,,\u795e\u79d8,Bi an nhap,,draft\n"
        "m-0003,2000,3000,,\u8df3\u8fc7,Bo qua,,skip\n",
        encoding="utf-8-sig",
    )
    (job / "translation" / "translated.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "m-0004",
                    "start_ms": 3000,
                    "end_ms": 4000,
                    "speaker": None,
                    "text_cn": "\u5927\u738b",
                    "text_vi": "Dai vuong",
                    "context_note": "",
                    "status": "draft",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (job / "context" / "characters.json").write_text(
        json.dumps([{"name_cn": "\u5c0f\u660e", "name_vi": "Tieu Minh", "source": "manual"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (job / "context" / "glossary.json").write_text(
        json.dumps({"\u795e\u79d8": "than bi"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (job / "context" / "reference_context.json").write_text(
        json.dumps(
            {
                "Names.txt": [{"source": "\u5982\u6765\u4f5b", "target": "Phat Nhu Lai"}],
                "VietPhrase.txt": [{"source": "\u9662\u5b50", "target": "san"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = collect_system_memory(jobs_dir)

    assert [item.text_cn for item in memory.translation_examples] == [
        "\u4f60\u597d",
        "\u795e\u79d8",
        "\u5927\u738b",
    ]
    assert memory.translation_examples[0].confidence == 0.95
    assert {item.name_cn for item in memory.characters} == {"\u5c0f\u660e", "\u5982\u6765\u4f5b"}
    assert {item.source_text for item in memory.glossary} == {"\u795e\u79d8", "\u9662\u5b50"}
    assert memory.warnings == []


def test_collect_system_memory_prioritizes_reviewed_rows(tmp_path):
    jobs_dir = tmp_path / "jobs"
    first = jobs_dir / "first"
    second = jobs_dir / "second"
    (first / "translation").mkdir(parents=True)
    (second / "translation").mkdir(parents=True)
    (first / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Chao ban ban nhap,,draft\n",
        encoding="utf-8-sig",
    )
    (second / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0002,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )

    memory = collect_system_memory(jobs_dir)

    assert len(memory.translation_examples) == 1
    assert memory.translation_examples[0].text_vi == "Xin chao"
    assert memory.translation_examples[0].confidence == 0.95
    assert memory.translation_examples[0].source_job == "second"


def test_collect_system_memory_excludes_current_job(tmp_path):
    jobs_dir = tmp_path / "jobs"
    current = jobs_dir / "current"
    old = jobs_dir / "old"
    (current / "translation").mkdir(parents=True)
    (old / "translation").mkdir(parents=True)
    (current / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u5f53\u524d,Current,,reviewed\n",
        encoding="utf-8-sig",
    )
    (old / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0002,0,1000,,\u5386\u53f2,History,,reviewed\n",
        encoding="utf-8-sig",
    )

    memory = collect_system_memory(jobs_dir, current_job=current)

    assert [item.text_cn for item in memory.translation_examples] == ["\u5386\u53f2"]


def test_collect_system_memory_records_json_warning_and_continues(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "broken"
    (job / "translation").mkdir(parents=True)
    (job / "context").mkdir()
    (job / "translation" / "review.csv").write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job / "context" / "characters.json").write_text("{bad json", encoding="utf-8")

    memory = collect_system_memory(jobs_dir)

    assert [item.text_cn for item in memory.translation_examples] == ["\u4f60\u597d"]
    assert len(memory.warnings) == 1
    assert "characters.json" in memory.warnings[0].path
    assert "Invalid JSON" in memory.warnings[0].message
```

- [ ] **Step 2: Run the scanner tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_memory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vietdub.memory'`.

- [ ] **Step 3: Implement the scanner**

Create `src/vietdub/memory.py` with this content:

```python
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
```

- [ ] **Step 4: Run the scanner tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_memory.py -v
```

Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit the scanner**

Run:

```bash
git add src/vietdub/memory.py tests/test_memory.py
git commit -m "feat: collect system memory from prior jobs"
```

## Task 2: Add Relevance Filtering

**Files:**
- Modify: `src/vietdub/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Add failing relevance tests**

Append this content to `tests/test_memory.py`:

```python
from vietdub.memory import (
    MemoryCharacter,
    MemoryGlossaryEntry,
    SystemMemory,
    TranslationExample,
    select_relevant_memory,
)
from vietdub.models import TimedSegment


def test_select_relevant_memory_keeps_only_current_transcript_matches():
    memory = SystemMemory(
        translation_examples=[
            TranslationExample(
                text_cn="\u4f60\u597d",
                text_vi="Xin chao",
                source_job="old",
                source="translation/review.csv",
                confidence=0.95,
            ),
            TranslationExample(
                text_cn="\u5b8c\u5168\u4e0d\u76f8\u5173",
                text_vi="Khong lien quan",
                source_job="old",
                source="translation/review.csv",
                confidence=0.95,
            ),
        ],
        characters=[
            MemoryCharacter(
                name_cn="\u5c0f\u660e",
                name_vi="Tieu Minh",
                source_job="old",
                source="context/characters.json",
                confidence=0.80,
            ),
            MemoryCharacter(
                name_cn="\u5927\u738b",
                name_vi="Dai vuong",
                source_job="old",
                source="context/characters.json",
                confidence=0.80,
            ),
        ],
        glossary=[
            MemoryGlossaryEntry(
                source_text="\u795e\u79d8",
                target="than bi",
                source_job="old",
                source="context/glossary.json",
                confidence=0.80,
            ),
            MemoryGlossaryEntry(
                source_text="\u9662\u5b50",
                target="san",
                source_job="old",
                source="context/glossary.json",
                confidence=0.80,
            ),
        ],
    )
    segments = [
        TimedSegment(
            id="m-0001",
            start_ms=0,
            end_ms=1000,
            text="\u5c0f\u660e\u8bf4\u4f60\u597d\uff0c\u771f\u795e\u79d8",
        )
    ]

    selected = select_relevant_memory(memory, segments)

    assert [item["text_cn"] for item in selected["translation_examples"]] == ["\u4f60\u597d"]
    assert [item["name_cn"] for item in selected["characters"]] == ["\u5c0f\u660e"]
    assert [item["source_text"] for item in selected["glossary"]] == ["\u795e\u79d8"]


def test_select_relevant_memory_limits_output_by_confidence_then_score():
    memory = SystemMemory(
        translation_examples=[
            TranslationExample(
                text_cn="\u4f60\u597d",
                text_vi="Xin chao",
                source_job="old",
                source="translation/review.csv",
                confidence=0.70,
            ),
            TranslationExample(
                text_cn="\u5c0f\u660e\u4f60\u597d",
                text_vi="Tieu Minh xin chao",
                source_job="old",
                source="translation/review.csv",
                confidence=0.95,
            ),
        ]
    )
    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u5c0f\u660e\u4f60\u597d")]

    selected = select_relevant_memory(memory, segments, max_examples=1)

    assert [item["text_cn"] for item in selected["translation_examples"]] == ["\u5c0f\u660e\u4f60\u597d"]
```

- [ ] **Step 2: Run the relevance tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_memory.py::test_select_relevant_memory_keeps_only_current_transcript_matches tests/test_memory.py::test_select_relevant_memory_limits_output_by_confidence_then_score -v
```

Expected: FAIL with `ImportError: cannot import name 'select_relevant_memory'`.

- [ ] **Step 3: Implement relevance selection**

Append this code to `src/vietdub/memory.py`:

```python

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
```

- [ ] **Step 4: Run memory tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_memory.py -v
```

Expected: PASS for all 6 tests.

- [ ] **Step 5: Commit relevance filtering**

Run:

```bash
git add src/vietdub/memory.py tests/test_memory.py
git commit -m "feat: filter system memory for current transcript"
```

## Task 3: Merge System Memory Into Context Bundle

**Files:**
- Modify: `src/vietdub/context.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Add failing context tests**

Replace `tests/test_context.py` with this content:

```python
from vietdub.context import build_context_bundle
from vietdub.models import DEFAULT_TONE, TimedSegment


def test_context_bundle_contains_episode_and_style():
    segments = [
        TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u5230\u5e95\u662f\u8c01"),
        TimedSegment(
            id="m-0002",
            start_ms=1000,
            end_ms=2000,
            text="\u90fd\u6000\u7591\u4f60\u6709\u4e00\u4e2a\u795e\u79d8\u7684\u8fc7\u5f80",
        ),
    ]

    bundle = build_context_bundle(segments, series_context={})

    assert "episode_context" in bundle
    assert "style_guide" in bundle
    assert bundle["style_guide"]["tone"] == DEFAULT_TONE
    assert bundle["scene_context"][0]["segment_ids"] == ["m-0001", "m-0002"]


def test_context_bundle_includes_reference_context():
    reference_context = {
        "Pronouns.txt": [{"source": "\u4f60", "target": "nguoi"}],
        "VietPhrase.txt": [{"source": "\u795e\u79d8", "target": "than bi"}],
    }

    bundle = build_context_bundle(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u6709\u795e\u79d8\u8fc7\u53bb")],
        series_context={},
        reference_context=reference_context,
    )

    assert bundle["reference_context"] == reference_context
    assert any("reference_context" in rule for rule in bundle["style_guide"]["translation_rules"])


def test_context_bundle_derives_characters_and_glossary_from_reference_context():
    reference_context = {
        "Names.txt": [{"source": "\u5982\u6765\u4f5b", "target": "Phat Nhu Lai"}],
        "Pronouns.txt": [{"source": "\u4f60", "target": "nguoi"}],
        "VietPhrase.txt": [{"source": "\u795e\u79d8", "target": "than bi"}],
    }

    bundle = build_context_bundle(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u5982\u6765\u4f5b\u4f60\u6709\u795e\u79d8")],
        series_context={},
        reference_context=reference_context,
    )

    assert bundle["characters"] == [
        {
            "name_cn": "\u5982\u6765\u4f5b",
            "name_vi": "Phat Nhu Lai",
            "source": "Names.txt",
        }
    ]
    assert bundle["glossary"] == {
        "\u4f60": "nguoi",
        "\u795e\u79d8": "than bi",
    }


def test_context_bundle_includes_system_memory_and_translation_examples():
    system_memory = {
        "translation_examples": [
            {
                "text_cn": "\u4f60\u597d",
                "text_vi": "Xin chao",
                "source_job": "old",
                "source": "translation/review.csv",
                "confidence": 0.95,
                "segment_id": "m-0001",
            }
        ],
        "characters": [
            {
                "name_cn": "\u5c0f\u660e",
                "name_vi": "Tieu Minh",
                "source_job": "old",
                "source": "context/characters.json",
                "confidence": 0.80,
            }
        ],
        "glossary": [
            {
                "source_text": "\u795e\u79d8",
                "target": "than bi tu lich su",
                "source_job": "old",
                "source": "context/glossary.json",
                "confidence": 0.80,
            }
        ],
    }

    bundle = build_context_bundle(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u5c0f\u660e\u4f60\u597d\u795e\u79d8")],
        series_context={},
        system_memory=system_memory,
    )

    assert bundle["system_memory"] == system_memory
    assert bundle["translation_examples"] == system_memory["translation_examples"]
    assert {
        "name_cn": "\u5c0f\u660e",
        "name_vi": "Tieu Minh",
        "source": "system_memory",
        "confidence": 0.80,
    } in bundle["characters"]
    assert bundle["glossary"]["\u795e\u79d8"] == "than bi tu lich su"
    assert any("system_memory" in rule for rule in bundle["style_guide"]["translation_rules"])


def test_context_bundle_keeps_reference_glossary_before_system_memory():
    reference_context = {
        "VietPhrase.txt": [{"source": "\u795e\u79d8", "target": "than bi tu data"}],
    }
    system_memory = {
        "translation_examples": [],
        "characters": [],
        "glossary": [
            {
                "source_text": "\u795e\u79d8",
                "target": "than bi tu lich su",
                "source_job": "old",
                "source": "context/glossary.json",
                "confidence": 0.80,
            }
        ],
    }

    bundle = build_context_bundle(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u795e\u79d8")],
        series_context={},
        reference_context=reference_context,
        system_memory=system_memory,
    )

    assert bundle["glossary"]["\u795e\u79d8"] == "than bi tu data"
```

- [ ] **Step 2: Run context tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_context.py -v
```

Expected: FAIL with `TypeError: build_context_bundle() got an unexpected keyword argument 'system_memory'`.

- [ ] **Step 3: Implement context merging**

Replace `src/vietdub/context.py` with this content:

```python
from __future__ import annotations

from .models import DEFAULT_TONE, TimedSegment


EPISODE_SUMMARY = (
    "Tap phim hoat hinh Trung Quoc ngan, thoai nhanh, "
    "co yeu to hai sa dieu."
)
SCENE_SUMMARY = (
    "Canh mo dau hoac doan thoai lien tuc can dich theo cung ngu canh."
)


def derive_characters(reference_context: dict) -> list[dict[str, str]]:
    return [
        {
            "name_cn": entry["source"],
            "name_vi": entry["target"].split("/")[0],
            "source": "Names.txt",
        }
        for entry in reference_context.get("Names.txt", [])
    ]


def derive_glossary(reference_context: dict) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for filename, entries in reference_context.items():
        if filename == "Names.txt":
            continue
        for entry in entries:
            glossary[entry["source"]] = entry["target"]
    return glossary


def build_context_bundle(
    segments: list[TimedSegment],
    series_context: dict,
    reference_context: dict | None = None,
    system_memory: dict | None = None,
) -> dict:
    reference_context = reference_context or {}
    system_memory = system_memory or {"translation_examples": [], "characters": [], "glossary": []}
    joined = " ".join(segment.text for segment in segments[:20])
    scene_ids = [segment.id for segment in segments[:20]]
    characters = merge_characters(derive_characters(reference_context), system_memory.get("characters", []))
    glossary = merge_glossary(derive_glossary(reference_context), system_memory.get("glossary", []))
    return {
        "series_context": series_context,
        "reference_context": reference_context,
        "system_memory": system_memory,
        "translation_examples": system_memory.get("translation_examples", []),
        "episode_context": {
            "summary": EPISODE_SUMMARY,
            "source_excerpt": joined,
        },
        "characters": characters,
        "glossary": glossary,
        "style_guide": {
            "tone": DEFAULT_TONE,
            "translation_rules": [
                "Uu tien cau thoai tu nhien hon dich sat chu.",
                "Giu punchline ngan de hop timing TTS.",
                "Dich nhat quan ten rieng va cach xung ho trong toan bo job.",
                "Use reference_context for names, pronouns, and phrase hints, but keep Vietnamese dialogue natural.",
                "Prefer high-confidence system_memory examples when they match the current line, especially reviewed review.csv rows.",
            ],
        },
        "scene_context": [
            {
                "segment_ids": scene_ids,
                "summary": SCENE_SUMMARY,
                "characters": [character["name_vi"] for character in characters],
                "tone": DEFAULT_TONE,
            }
        ],
    }


def merge_characters(reference_characters: list[dict[str, str]], memory_characters: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for character in reference_characters:
        merged[character["name_cn"]] = character
    for character in memory_characters:
        name_cn = str(character.get("name_cn") or "").strip()
        name_vi = str(character.get("name_vi") or "").strip()
        if not name_cn or not name_vi or name_cn in merged:
            continue
        merged[name_cn] = {
            "name_cn": name_cn,
            "name_vi": name_vi,
            "source": "system_memory",
            "confidence": float(character.get("confidence") or 0.0),
        }
    return list(merged.values())


def merge_glossary(reference_glossary: dict[str, str], memory_glossary: list[dict]) -> dict[str, str]:
    merged = dict(reference_glossary)
    for entry in memory_glossary:
        source_text = str(entry.get("source_text") or "").strip()
        target = str(entry.get("target") or "").strip()
        if not source_text or not target or source_text in merged:
            continue
        merged[source_text] = target
    return merged
```

- [ ] **Step 4: Run context tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_context.py -v
```

Expected: PASS for all 5 tests.

- [ ] **Step 5: Commit context integration**

Run:

```bash
git add src/vietdub/context.py tests/test_context.py
git commit -m "feat: merge system memory into context bundle"
```

## Task 4: Wire System Memory Into Review Pipeline

**Files:**
- Modify: `src/vietdub/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline integration test**

Append this content to `tests/test_pipeline.py`:

```python

def test_review_pipeline_writes_selected_system_memory(monkeypatch, tmp_path):
    from vietdub.memory import MemoryWarningItem, SystemMemory, TranslationExample
    from vietdub.pipeline import run_review_pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    captured_context = {}

    def fake_extract_audio(video_path, audio_path, sample_rate):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")

    class FakeSttEngine:
        def __init__(self, model, language):
            self.model = model
            self.language = language

        def transcribe(self, audio_path):
            return [
                TimedSegment(
                    id="s-0001",
                    start_ms=0,
                    end_ms=1000,
                    text="\u5c0f\u660e\u8bf4\u4f60\u597d",
                    source="stt",
                )
            ]

    class FakeOcrEngine:
        def recognize(self, video_path):
            return [
                TimedSegment(
                    id="o-0001",
                    start_ms=0,
                    end_ms=1000,
                    text="\u5c0f\u660e\u8bf4\u4f60\u597d",
                    source="ocr",
                )
            ]

    def fake_translate(segments, context_bundle, api_key, model, base_url):
        captured_context.update(context_bundle)
        return [
            type(
                "Row",
                (),
                {
                    "segment_id": "m-0001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker": None,
                    "text_cn": "\u5c0f\u660e\u8bf4\u4f60\u597d",
                    "text_vi": "Tieu Minh noi xin chao",
                    "context_note": "",
                    "status": "draft",
                    "model_dump": lambda self: {
                        "segment_id": self.segment_id,
                        "start_ms": self.start_ms,
                        "end_ms": self.end_ms,
                        "speaker": self.speaker,
                        "text_cn": self.text_cn,
                        "text_vi": self.text_vi,
                        "context_note": self.context_note,
                        "status": self.status,
                    },
                },
            )()
        ]

    def fake_collect_system_memory(jobs_dir, current_job=None):
        return SystemMemory(
            translation_examples=[
                TranslationExample(
                    text_cn="\u4f60\u597d",
                    text_vi="Xin chao",
                    source_job="old",
                    source="translation/review.csv",
                    confidence=0.95,
                )
            ],
            warnings=[MemoryWarningItem(path="old/context/characters.json", message="Invalid JSON: test")],
        )

    monkeypatch.setattr("vietdub.media.extract_audio", fake_extract_audio)
    monkeypatch.setattr("vietdub.stt.FasterWhisperSttEngine", FakeSttEngine)
    monkeypatch.setattr("vietdub.ocr.PaddleSubtitleOcrEngine", FakeOcrEngine)
    monkeypatch.setattr("vietdub.reference.build_reference_context", lambda segments, data_dir: {})
    monkeypatch.setattr("vietdub.translate.translate_with_llm", fake_translate)
    monkeypatch.setattr("vietdub.memory.collect_system_memory", fake_collect_system_memory)

    settings = type(
        "Settings",
        (),
        {
            "sample_rate": 44100,
            "stt_model": "tiny",
            "stt_language": "zh",
            "reference_data_dir": str(tmp_path / "data"),
            "openai_api_key": "key",
            "llm_model": "model",
            "openai_base_url": "",
        },
    )()

    job = run_review_pipeline(video=video, jobs_dir=tmp_path / "jobs", series=None, settings=settings)

    assert captured_context["translation_examples"][0]["text_vi"] == "Xin chao"
    assert (job.root / "context" / "system_memory.json").exists()
    assert (job.root / "context" / "translation_examples.json").exists()
    warnings = json.loads((job.root / "context" / "system_memory_warnings.json").read_text(encoding="utf-8"))
    assert warnings == [{"path": "old/context/characters.json", "message": "Invalid JSON: test"}]
```

- [ ] **Step 2: Run the pipeline integration test and verify it fails**

Run:

```powershell
python -m pytest tests/test_pipeline.py::test_review_pipeline_writes_selected_system_memory -v
```

Expected: FAIL because `context/system_memory.json` is not written and `captured_context` has no `translation_examples`.

- [ ] **Step 3: Implement pipeline wiring**

In `src/vietdub/pipeline.py`, replace the body of `run_review_pipeline` with this code:

```python
def run_review_pipeline(video: Path, jobs_dir: Path, series: str | None, settings) -> Job:
    from .config import Settings
    from .media import extract_audio
    from .memory import collect_system_memory, select_relevant_memory
    from .ocr import PaddleSubtitleOcrEngine
    from .reference import build_reference_context
    from .stt import FasterWhisperSttEngine
    from .translate import translate_with_llm

    typed_settings: Settings = settings
    job = JobManager(jobs_dir).create(video, series=series)
    audio_path = job.root / "audio" / "original.wav"
    extract_audio(job.input_video, audio_path, typed_settings.sample_rate)
    job.mark_done(StepName.EXTRACT, {"audio": str(audio_path)})

    stt_segments = FasterWhisperSttEngine(typed_settings.stt_model, typed_settings.stt_language).transcribe(audio_path)
    ocr_segments = PaddleSubtitleOcrEngine().recognize(job.input_video)
    merged = merge_segments(stt_segments, ocr_segments)
    reference_context = build_reference_context(merged, Path(typed_settings.reference_data_dir))
    raw_system_memory = collect_system_memory(jobs_dir, current_job=job.root)
    selected_system_memory = select_relevant_memory(raw_system_memory, merged)
    context_bundle = build_context_bundle(
        merged,
        series_context={},
        reference_context=reference_context,
        system_memory=selected_system_memory,
    )
    translations = translate_with_llm(
        merged,
        context_bundle,
        typed_settings.openai_api_key,
        typed_settings.llm_model,
        typed_settings.openai_base_url,
    )

    write_segments(job, "stt/segments.json", stt_segments)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    if raw_system_memory.warnings:
        job.write_json("context/system_memory_warnings.json", [warning.model_dump() for warning in raw_system_memory.warnings])
    job.write_json("translation/translated.json", [row.model_dump() for row in translations])
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    return job
```

- [ ] **Step 4: Run pipeline tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: PASS for all pipeline tests.

- [ ] **Step 5: Run context and memory regression tests**

Run:

```powershell
python -m pytest tests/test_memory.py tests/test_context.py tests/test_pipeline.py -v
```

Expected: PASS for all selected tests.

- [ ] **Step 6: Commit pipeline wiring**

Run:

```bash
git add src/vietdub/pipeline.py tests/test_pipeline.py
git commit -m "feat: use system memory during review pipeline"
```

## Task 5: Document System Memory Artifacts

**Files:**
- Modify: `README.md`
- Modify: `docs/manual-test.md`

- [ ] **Step 1: Update README**

Append this section to `README.md`:

```markdown

## System Memory

Each review run scans previous job folders under `jobs/` before translation. It reuses:

- `translation/review.csv`
- `translation/translated.json`
- `context/characters.json`
- `context/glossary.json`
- `context/reference_context.json`

The current job is excluded from the scan. Reviewed rows in `review.csv` have the highest confidence, draft rows with `text_vi` are lower confidence, and generated `translated.json` rows are fallback examples.

New jobs write these memory artifacts under `jobs/<job-name>/context/`:

- `system_memory.json`: selected historical examples, characters, and glossary entries relevant to the current transcript
- `translation_examples.json`: high-confidence Chinese-to-Vietnamese examples included in the LLM prompt
- `characters.json`: names merged from reference data and relevant system memory
- `glossary.json`: terms merged from reference data and relevant system memory
- `system_memory_warnings.json`: scanner warnings when old job files are malformed
```

- [ ] **Step 2: Update manual test documentation**

Replace the `## Acceptance Criteria` section in `docs/manual-test.md` with this content:

```markdown
## Acceptance Criteria

- A job folder is created under `jobs`.
- `stt/segments.json`, `ocr/subtitles.json`, `transcript/merged.json`, and `context/style_guide.json` exist after review run.
- `context/system_memory.json` exists after review run.
- `context/translation_examples.json` exists after review run.
- `context/characters.json` and `context/glossary.json` contain JSON that can be opened in a text editor.
- `translation/review.csv` opens in Excel without mojibake.
- `output/subtitles_vi.srt` contains the reviewed Vietnamese lines after `vietdub resume <job> --from tts` or `vietdub run <video> --mode auto`.
- `tts/segments` contains one MP3 per reviewed non-empty row when Edge TTS succeeds.
```

- [ ] **Step 3: Run docs-adjacent regression tests**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_pipeline.py -v
```

Expected: PASS for all selected tests.

- [ ] **Step 4: Commit documentation**

Run:

```bash
git add README.md docs/manual-test.md
git commit -m "docs: describe system memory artifacts"
```

## Task 6: Full Regression And Manual Smoke Check

**Files:**
- Verify: full repository test suite
- Verify: one lightweight CLI run through monkeypatched/unit coverage already exists

- [ ] **Step 1: Run the full unit suite**

Run:

```powershell
python -m pytest -v
```

Expected: PASS for all tests.

- [ ] **Step 2: Inspect the changed files**

Run:

```powershell
git diff --stat HEAD~5..HEAD
```

Expected: output includes these files:

```text
README.md
docs/manual-test.md
src/vietdub/context.py
src/vietdub/memory.py
src/vietdub/pipeline.py
tests/test_context.py
tests/test_memory.py
tests/test_pipeline.py
```

- [ ] **Step 3: Confirm ignored data and jobs stay uncommitted**

Run:

```powershell
git status --short
```

Expected: no tracked changes remain from implementation commits. It is acceptable for ignored or unrelated untracked local folders such as `data/`, `jobs/`, or `docs/superpowers/` to remain visible depending on local `.gitignore` and existing workspace state.

## Self-Review Notes

- Spec coverage: The plan adds historical scanning from `review.csv`, `translated.json`, and `context/*.json`; ranks sources by confidence; excludes the current job; filters memory to current transcript; writes `system_memory.json`, `translation_examples.json`, `characters.json`, `glossary.json`, and warning artifacts; and documents how to inspect them.
- Placeholder scan: The plan contains concrete file paths, commands, expected failures, expected passes, and complete code blocks for all code changes.
- Type consistency: The scanner exposes `SystemMemory`, `TranslationExample`, `MemoryCharacter`, `MemoryGlossaryEntry`, `MemoryWarningItem`, `collect_system_memory()`, and `select_relevant_memory()`. `build_context_bundle()` accepts a plain selected-memory dict to keep pipeline JSON writing simple.
