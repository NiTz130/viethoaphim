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


def test_collect_system_memory_skips_character_with_bad_confidence(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "old"
    (job / "context").mkdir(parents=True)
    (job / "context" / "characters.json").write_text(
        json.dumps(
            [
                {"name_cn": "\\u574f", "name_vi": "Hong", "confidence": "bad"},
                {"name_cn": "\\u597d", "name_vi": "Tot", "confidence": 0.9},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = collect_system_memory(jobs_dir)

    assert [item.name_cn for item in memory.characters] == ["\\u597d"]
    assert len(memory.warnings) == 1
    assert "characters.json" in memory.warnings[0].path
    assert "confidence" in memory.warnings[0].message


def test_collect_system_memory_skips_glossary_entry_with_bad_confidence(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job = jobs_dir / "old"
    (job / "context").mkdir(parents=True)
    (job / "context" / "glossary.json").write_text(
        json.dumps(
            {
                "\\u574f": {"target": "hong", "confidence": "bad"},
                "\\u597d": {"target": "tot", "confidence": 0.9},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = collect_system_memory(jobs_dir)

    assert [item.source_text for item in memory.glossary] == ["\\u597d"]
    assert len(memory.warnings) == 1
    assert "glossary.json" in memory.warnings[0].path
    assert "confidence" in memory.warnings[0].message


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


def test_memory_surfaces_corrupt_utf8_with_replace_marker(tmp_path):
    """M5 fix: corrupt UTF-8 bytes are replaced with ? instead of silently
    dropped, so the corruption is visible to the operator."""
    from vietdub.memory import _collect_review_csv, MemoryWarningItem

    job_dir = tmp_path / "job"
    (job_dir / "translation").mkdir(parents=True)
    # Write a review CSV with one valid line and one line containing a
    # corrupt UTF-8 byte (0x80 is not valid UTF-8).
    csv_bytes = (
        b"segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        b"m-0001,0,1000,,\xe4\xbd\xa0\xe5\xa5\xbd,Xin ch\xc3\xa0o,,reviewed\n"
        b"m-0002,1000,2000,,\x80\x81\x82,Xin chao,,reviewed\n"
    )
    (job_dir / "translation" / "review.csv").write_bytes(csv_bytes)

    examples: dict = {}
    warnings: list[MemoryWarningItem] = []
    _collect_review_csv(job_dir, examples, warnings)

    # Two rows should be loaded (the corrupt one is recovered with U+FFFD).
    assert len(examples) == 2
    # The corrupt row's text_cn should contain U+FFFD replacement markers.
    # Note: examples is keyed by text_cn, not segment_id; iterate values.
    assert any("�" in ex.text_cn for ex in examples.values())


def test_load_json_distinguishes_missing_from_corrupt(tmp_path):
    """L5: _load_json returns (None, None) for missing files and (None, warning)
    for corrupt files — caller can distinguish the two cases."""
    from vietdub.memory import _load_json, MemoryWarningItem

    missing = tmp_path / "does-not-exist.json"
    data, warning = _load_json(missing)
    assert data is None
    assert warning is None

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    data, warning = _load_json(corrupt)
    assert data is None
    assert warning is not None
    assert isinstance(warning, MemoryWarningItem)
    assert "Invalid JSON" in warning.message

    good = tmp_path / "good.json"
    good.write_text('{"k": "v"}', encoding="utf-8")
    data, warning = _load_json(good)
    assert data == {"k": "v"}
    assert warning is None
