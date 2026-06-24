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
