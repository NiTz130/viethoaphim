import json

import pytest

from vietdub.models import TimedSegment, TranslationRow
from vietdub.translate import (
    build_translation_prompt,
    export_review_csv,
    import_review_csv,
    parse_translation_response,
    translate_with_llm,
)


def test_review_csv_round_trip(tmp_path):
    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d", speaker="\u7537")]
    rows = [
        TranslationRow(
            segment_id="m-0001",
            start_ms=0,
            end_ms=1000,
            speaker="\u7537",
            text_cn="\u4f60\u597d",
            text_vi="Ch\u00e0o nha",
        )
    ]
    path = tmp_path / "review.csv"

    export_review_csv(path, segments, rows)
    loaded = import_review_csv(path)

    assert loaded[0].segment_id == "m-0001"
    assert loaded[0].text_vi == "Ch\u00e0o nha"


def test_export_review_csv_preserves_segment_timing_and_source_text(tmp_path):
    segments = [
        TimedSegment(
            id="m-0001",
            start_ms=0,
            end_ms=1000,
            text="\u4f60\u597d",
            speaker="\u7537",
        )
    ]
    rows = [
        TranslationRow(
            segment_id="m-0001",
            start_ms=9999,
            end_ms=12000,
            speaker="\u5973",
            text_cn="\u9519\u8bef",
            text_vi="Chao ban",
            context_note="giu giong hai",
            status="reviewed",
        )
    ]
    path = tmp_path / "review.csv"

    export_review_csv(path, segments, rows)
    loaded = import_review_csv(path)

    assert loaded == [
        TranslationRow(
            segment_id="m-0001",
            start_ms=0,
            end_ms=1000,
            speaker="\u7537",
            text_cn="\u4f60\u597d",
            text_vi="Chao ban",
            context_note="giu giong hai",
            status="reviewed",
        )
    ]


def test_translation_prompt_explicitly_requests_json_output():
    prompt = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d")],
        context_bundle={},
    )

    assert "Return JSON only" in prompt
    assert "translations" in prompt


def test_parse_translation_response_rejects_invalid_json():
    with pytest.raises(RuntimeError, match="Invalid LLM translation response"):
        parse_translation_response("{bad json")


def test_parse_translation_response_requires_translations_list():
    with pytest.raises(RuntimeError, match="translations"):
        parse_translation_response('{"items": []}')


def test_parse_translation_response_identifies_bad_item():
    with pytest.raises(RuntimeError, match="translation item 1"):
        parse_translation_response('{"translations": ["not an object"]}')


def test_parse_translation_response_warns_and_coerces_invalid_status(caplog):
    """H4 fix: LLM returns an invalid status; we log a warning AND coerce to 'draft'."""
    payload = json.dumps(
        {
            "translations": [
                {
                    "segment_id": "m-0001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker": None,
                    "text_cn": "你好",
                    "text_vi": "Xin chào",
                    "context_note": "",
                    "status": "published",  # invalid
                }
            ]
        }
    )

    rows = parse_translation_response(payload)

    assert rows[0].status == "draft"
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert warnings, "Expected at least one WARNING log from vietdub.translate"
    msg = warnings[0].getMessage()
    assert "invalid status" in msg
    assert "m-0001" in msg


def _valid_translations_payload():
    return {
        "translations": [
            {
                "segment_id": "m-0001",
                "start_ms": 0,
                "end_ms": 1000,
                "speaker": None,
                "text_cn": "你好",
                "text_vi": "Xin chào",
                "context_note": "",
                "status": "draft",
            }
        ]
    }


def test_parse_translation_response_strips_markdown_fences():
    """LLMs (notably MiniMax M3) sometimes wrap JSON in ```json ... ``` fences."""
    payload = json.dumps(_valid_translations_payload())
    fenced = "```json\n" + payload + "\n```"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_handles_single_line_fence():
    payload = json.dumps(_valid_translations_payload())
    # No newline between ```json and {
    fenced = "```json " + payload + "\n```"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_handles_trailing_only_fence():
    payload = json.dumps(_valid_translations_payload())
    # No opening fence, only trailing ```
    fenced = payload + "\n```"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_handles_crlf_fences():
    payload = json.dumps(_valid_translations_payload())
    fenced = "```json\r\n" + payload + "\r\n```\r\n"

    rows = parse_translation_response(fenced)

    assert len(rows) == 1
    assert rows[0].segment_id == "m-0001"


def test_parse_translation_response_preserves_raw_error_context():
    with pytest.raises(RuntimeError, match="Invalid LLM translation response") as exc_info:
        parse_translation_response("Not valid JSON at all")

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("Raw (pre-strip) JSON also failed" in n for n in notes)


def test_import_review_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("segment_id,start_ms,end_ms,text_cn,text_vi\nm-0001,0,1000,\u4f60\u597d,Xin chao\n", encoding="utf-8-sig")

    with pytest.raises(RuntimeError, match="missing columns"):
        import_review_csv(path)


def test_import_review_csv_reports_invalid_row_number(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,1000,0,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(RuntimeError, match="row 2"):
        import_review_csv(path)


def test_build_translation_prompt_includes_examples():
    """build_translation_prompt payload contains 3+ few-shot examples."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    assert "examples" in payload
    assert len(payload["examples"]) >= 3
    for ex in payload["examples"]:
        assert "segment_id" in ex
        assert "text_cn" in ex
        assert "text_vi" in ex


def test_build_translation_prompt_includes_length_targets():
    """Each segment dict has target_vi_chars based on duration."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=5000, text="\u6d4b\u8bd5")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    seg = payload["segments"][0]
    assert "target_vi_chars" in seg
    assert seg["target_vi_chars"] == 77  # 5s * 14 chars/sec * 1.1 buffer \u2248 77


def test_length_target_vi_chars_minimum_floor():
    """Very short segments still get a minimum target of 10 chars."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=100, text="hi")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    seg = payload["segments"][0]
    assert seg["target_vi_chars"] >= 10


def test_translation_examples_have_valid_format():
    """All examples have segment_id prefix 'ex-' and non-empty text fields."""
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d")],
        context_bundle={},
    )
    payload = json.loads(prompt_json)
    for ex in payload["examples"]:
        assert ex["segment_id"].startswith("ex-")
        assert ex["text_cn"]
        assert ex["text_vi"]


def test_translate_pipeline_produces_valid_response():
    """Integration: new prompts don't break the existing parse path."""
    import sys
    import types

    from vietdub.models import TimedSegment

    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text
            self.type = "text"

    class _FakeMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "\u4f60\u597d",
                        "text_vi": "Xin ch\u00e0o",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            self.messages = _FakeMessages()

    def _settings(**overrides):
        base = {
            "anthropic_api_key": "test-key",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "llm_model": "MiniMax-M3",
        }
        base.update(overrides)
        return types.SimpleNamespace(**base)

    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d")],
        context_bundle={},
        settings=_settings(),
    )

    assert len(rows) == 1
    assert rows[0].text_vi == "Xin ch\u00e0o"
    monkey.undo()


def test_parse_translation_response_warns_on_oversized_translation(caplog):
    """Translation > 120% of target emits [LEN WARNING] to the logger."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0042",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": None,
            "text_cn": "\u4f60\u597d",
            "text_vi": "\u554a" * 50,
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert warnings, "Expected at least one WARNING log from vietdub.translate"
    msg = warnings[0].getMessage()
    assert "[LEN WARNING]" in msg
    assert "m-0042" in msg
    assert "over" in msg


def test_parse_translation_response_warns_on_undersized_translation(caplog):
    """Translation < 80% of target emits [LEN WARNING] to the logger."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0099",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": None,
            "text_cn": "\u4f60\u597d\u4e16\u754c",
            "text_vi": "xin ch\u00e0o",
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert warnings, "Expected at least one WARNING log from vietdub.translate"
    msg = warnings[0].getMessage()
    assert "[LEN WARNING]" in msg
    assert "m-0099" in msg
    assert "under" in msg


def test_parse_translation_response_no_warn_within_tolerance(caplog):
    """Translation within \u00b120% of target emits no [LEN WARNING]."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-0001",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": None,
            "text_cn": "\u4f60\u597d",
            "text_vi": "xin ch\u00e0o b\u1ea1n",
            "context_note": "",
            "status": "draft",
        }]
    })
    parse_translation_response(payload)
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert not [w for w in warnings if "[LEN WARNING]" in w.getMessage()], (
        f"Expected no [LEN WARNING] within tolerance, got: {[w.getMessage() for w in warnings]}"
    )


def test_parse_translation_response_skips_length_check_for_empty_translation(caplog):
    """Empty text_vi is not length-checked (would always 'under')."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-empty",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": None,
            "text_cn": "\u4f60\u597d",
            "text_vi": "",
            "context_note": "",
            "status": "draft",
        }]
    })
    rows = parse_translation_response(payload)
    assert len(rows) == 1
    assert rows[0].text_vi == ""
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert not [w for w in warnings if "[LEN WARNING]" in w.getMessage()]


def test_parse_translation_response_skips_length_check_for_very_short_segments(caplog):
    """Segments with target < 10 chars (floor region) skip length check."""
    payload = json.dumps({
        "translations": [{
            "segment_id": "m-short",
            "start_ms": 0,
            "end_ms": 100,
            "speaker": None,
            "text_cn": "hi",
            "text_vi": "a" * 50,
            "context_note": "",
            "status": "draft",
        }]
    })
    rows = parse_translation_response(payload)
    assert len(rows) == 1
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert not [w for w in warnings if "[LEN WARNING]" in w.getMessage()]


def test_extract_terms_from_translations_finds_chinese_names():
    """Heuristic extracts Chinese names → Vietnamese capitalized words."""
    from vietdub.translate import _update_glossary
    from vietdub.models import TranslationRow

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="长孙无忌 is here", text_vi="Trưởng Tôn đang ở đây",
            context_note="", status="draft",
        )
    ]
    batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="长孙无忌 is here")]
    glossary: dict = {}
    _update_glossary(glossary, rows, batch)
    assert "长孙无忌" in glossary
    assert glossary["长孙无忌"] == "Trưởng"


def test_extract_terms_from_translations_skips_non_capitalized():
    """Lowercase Vietnamese words don't get added to glossary."""
    from vietdub.translate import _update_glossary
    from vietdub.models import TranslationRow

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好世界", text_vi="xin chào bạn",
            context_note="", status="draft",
        )
    ]
    batch = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好世界")]
    glossary: dict = {}
    _update_glossary(glossary, rows, batch)
    assert glossary == {}


def test_translate_with_llm_passes_glossary_to_subsequent_batches(monkeypatch):
    """Glossary accumulates across batches and is passed forward."""
    import sys
    import types

    captured_prompts: list = []

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _CapturingMessages:
        """Returns a different translation each call to simulate cross-batch evolution."""
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            self.call_count += 1
            # First call returns 1 row with Chinese name "Trưởng"
            # Second call returns 1 row with Chinese name "Trưởng" again (consistency check)
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": f"m-{self.call_count:04d}",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "长孙无忌",
                        "text_vi": "Trưởng Tôn",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _CapturingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _CapturingMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_CapturingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    segments = [
        TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="长孙无忌"),
        TimedSegment(id="m-0002", start_ms=0, end_ms=1000, text="长孙无忌"),
    ]
    # With LLM_BATCH_SIZE=50, both fit in one batch. To force 2 batches,
    # we patch LLM_BATCH_SIZE temporarily.
    import vietdub.translate as t
    monkeypatch.setattr(t, "LLM_BATCH_SIZE", 1)
    translate_with_llm(segments=segments, context_bundle={}, settings=settings)

    # With review=True (default), each batch now makes 2 LLM calls (translate + review).
    # Filter to only translate prompts (those with "consistency_terms" — review prompts
    # have a different payload shape with no consistency_terms).
    translate_prompts = [
        p for p in captured_prompts
        if "consistency_terms" in json.loads(p)
    ]
    assert len(translate_prompts) == 2
    # First batch's prompt: empty consistency_terms (glossary starts empty if Names.txt absent)
    first_payload = json.loads(translate_prompts[0])
    second_payload = json.loads(translate_prompts[1])
    # Second batch's prompt should include the glossary entry from batch 1
    sources = [t["source"] for t in second_payload.get("consistency_terms", [])]
    assert "长孙无忌" in sources
    targets = [t["target"] for t in second_payload.get("consistency_terms", [])]
    # Heuristic maps first N Chinese names to first N Vietnamese capitalized words.
    # With vi_words = ["Trưởng", "Tôn", ...], only "Trưởng" is stored.
    assert "Trưởng" in targets


def test_cap_glossary_drops_oldest_entries():
    """When glossary exceeds 50 entries, oldest are dropped FIFO."""
    from vietdub.translate import _cap_glossary

    glossary = {f"name_{i}": f"translation_{i}" for i in range(60)}
    _cap_glossary(glossary, max_entries=50)
    assert len(glossary) == 50
    assert "name_0" not in glossary  # oldest dropped
    assert "name_59" in glossary    # newest kept


def test_build_translation_prompt_includes_consistency_terms():
    """build_translation_prompt includes glossary as consistency_terms."""
    glossary = {"长孙无忌": "Trưởng Tôn"}
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        glossary=glossary,
    )
    payload = json.loads(prompt_json)
    assert "consistency_terms" in payload
    assert {"source": "长孙无忌", "target": "Trưởng Tôn"} in payload["consistency_terms"]


def test_review_batch_for_length_returns_refined_translations(monkeypatch):
    """Review pass returns refined translations that match length target."""
    import sys
    import types
    from vietdub.translate import _review_batch_for_length

    # Build initial rows (oversized)
    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好世界", text_vi="Xin chào bạn ơi nhé nhé nhé",
            context_note="", status="draft",
        )
    ]

    # Mock anthropic to return a shorter text_vi
    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _RefiningMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "你好世界",
                        "text_vi": "Xin chào bạn",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _RefiningAnthropic:
        def __init__(self, **kwargs):
            self.messages = _RefiningMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_RefiningAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert len(refined) == 1
    assert refined[0].text_vi == "Xin chào bạn"
    # Original fields preserved
    assert refined[0].segment_id == "m-0001"
    assert refined[0].start_ms == 0
    assert refined[0].end_ms == 1000


def test_review_batch_for_length_preserves_unchanged_translations(monkeypatch):
    """Review pass preserves translations already within budget."""
    import sys
    import types
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(
            segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
            text_cn="你好", text_vi="Xin chào",  # short, within budget
            context_note="", status="draft",
        )
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns SAME text (review should preserve it)
    class _PreservingMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "你好",
                        "text_vi": "Xin chào",  # unchanged
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _PreservingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _PreservingMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_PreservingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "Xin chào"


def test_review_batch_for_length_handles_partial_review_response(monkeypatch):
    """If review returns only some segments, keep originals for missing ones."""
    import sys
    import types
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="a", text_vi="orig1", context_note="", status="draft"),
        TranslationRow(segment_id="m-0002", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="b", text_vi="orig2", context_note="", status="draft"),
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns only 1 of 2 segments
    class _PartialMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "a",
                        "text_vi": "refined1",
                        "context_note": "",
                        "status": "draft",
                    }]  # m-0002 missing
                }))],
                stop_reason="end_turn",
            )

    class _PartialAnthropic:
        def __init__(self, **kwargs):
            self.messages = _PartialMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_PartialAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "refined1"  # updated
    assert refined[1].text_vi == "orig2"     # kept original


def test_review_batch_for_length_skips_empty_refinement(monkeypatch):
    """If review returns empty text_vi, keep original."""
    import sys
    import types
    from vietdub.translate import _review_batch_for_length

    rows = [
        TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000, speaker=None,
                       text_cn="a", text_vi="original text", context_note="", status="draft"),
    ]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    # Mock returns EMPTY text_vi
    class _EmptyMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "a",
                        "text_vi": "",  # empty!
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _EmptyAnthropic:
        def __init__(self, **kwargs):
            self.messages = _EmptyMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_EmptyAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    refined = _review_batch_for_length(rows, context_bundle={}, settings=settings)
    assert refined[0].text_vi == "original text"  # kept


def test_translate_with_llm_runs_review_pass_when_enabled(monkeypatch):
    """translate_with_llm calls LLM twice per batch when review=True, once when False."""
    import sys
    import types

    import vietdub.translate as t
    call_count = [0]

    class _TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _CountingMessages:
        def create(self, **kwargs):
            call_count[0] += 1
            # Return a single-row response (works for both translate and review)
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [{
                        "segment_id": "m-0001",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker": None,
                        "text_cn": "你好",
                        "text_vi": "Xin chào",
                        "context_note": "",
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _CountingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _CountingMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_CountingAnthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.minimax.io/anthropic",
        llm_model="MiniMax-M3",
    )

    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")]

    # With review=True (default): expect 2 LLM calls (translate + review)
    call_count[0] = 0
    translate_with_llm(segments=segments, context_bundle={}, settings=settings, review=True)
    assert call_count[0] == 2, f"Expected 2 LLM calls (translate + review), got {call_count[0]}"

    # With review=False: expect 1 LLM call (translate only)
    call_count[0] = 0
    translate_with_llm(segments=segments, context_bundle={}, settings=settings, review=False)
    assert call_count[0] == 1, f"Expected 1 LLM call (translate only), got {call_count[0]}"


def test_build_translation_prompt_includes_pronoun_guide():
    """build_translation_prompt payload includes pronoun_guide section."""
    pronoun_guide = [{"source": "你自己", "target": "chính ngươi"}]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        pronoun_guide=pronoun_guide,
    )
    payload = json.loads(prompt_json)
    assert "pronoun_guide" in payload
    assert payload["pronoun_guide"] == pronoun_guide


def test_build_translation_prompt_includes_phrase_patterns():
    """build_translation_prompt payload includes phrase_patterns section with placeholders."""
    phrase_patterns = [{"source": "与{0}为敌为友", "target": "cùng {0} là địch là bạn"}]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        phrase_patterns=phrase_patterns,
    )
    payload = json.loads(prompt_json)
    assert "phrase_patterns" in payload
    assert payload["phrase_patterns"] == phrase_patterns
    # Placeholder preserved
    assert "{0}" in payload["phrase_patterns"][0]["source"]


def test_build_translation_prompt_includes_ignore_list():
    """build_translation_prompt payload includes ignore_list section."""
    ignore_list = ["( 小说 《 九 鼎记 ... )", "(未 完 待续)..."]
    prompt_json = build_translation_prompt(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="test")],
        context_bundle={},
        ignore_list=ignore_list,
    )
    payload = json.loads(prompt_json)
    assert "ignore_list" in payload
    assert payload["ignore_list"] == ignore_list


def test_load_helpers_resilient_to_missing_files(tmp_path):
    """All 3 loaders return empty list when file missing (no exception)."""
    from vietdub.translate import _load_pronouns, _load_phrase_patterns, _load_ignore_list

    # tmp_path doesn't have any of the data files
    assert _load_pronouns(tmp_path) == []
    assert _load_phrase_patterns(tmp_path) == []
    assert _load_ignore_list(tmp_path) == []


def test_translate_warns_and_falls_back_when_pronouns_unavailable(monkeypatch, caplog):
    """H2 fix: when reference data fails to load, we still get a translation
    but the user sees a Warning on stderr so the failure is diagnosable."""
    from vietdub import translate as translate_mod
    from vietdub.models import TimedSegment

    # Make the reference-data lookup raise.
    monkeypatch.setattr(
        translate_mod, "find_reference_data_dir",
        lambda _root: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    # Patch anthropic so translate_with_llm doesn't make a real call.
    from vietdub.translate import translate_with_llm
    import sys, types

    class _TextBlock:
        def __init__(self, text): self.text = text; self.type = "text"
    class _Messages:
        def __init__(self):
            self.calls = []
        def create(self, **kwargs):
            self.calls.append(kwargs)
            user_prompt = kwargs["messages"][0]["content"]
            payload = __import__("json").loads(user_prompt)
            segs = payload["segments"]
            # Handle both translation pass (id/text) and review pass (segment_id/text_cn).
            def _row(s):
                if "id" in s:
                    return {
                        "segment_id": s["id"],
                        "start_ms": s["start_ms"],
                        "end_ms": s["end_ms"],
                        "speaker": s.get("speaker"),
                        "text_cn": s["text"],
                        "text_vi": "fallback vi",
                        "context_note": "",
                        "status": "draft",
                    }
                return {
                    "segment_id": s["segment_id"],
                    "start_ms": s["start_ms"],
                    "end_ms": s["end_ms"],
                    "speaker": s.get("speaker"),
                    "text_cn": s.get("text_cn", ""),
                    "text_vi": s.get("text_vi", "fallback vi"),
                    "context_note": "",
                    "status": "draft",
                }
            return types.SimpleNamespace(
                content=[_TextBlock(__import__("json").dumps({
                    "translations": [_row(s) for s in segs]
                }))],
                stop_reason="end_turn",
            )
    class _Anthropic:
        def __init__(self, **kwargs): self.kwargs = kwargs; self.messages = _Messages()
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="k", anthropic_base_url="https://api.minimax.io/anthropic", llm_model="MiniMax-M3",
    )
    rows = translate_with_llm(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="hello")],
        {},
        settings=settings,
    )
    assert len(rows) == 1
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert warnings, "Expected at least one WARNING log from vietdub.translate"
    msg = warnings[0].getMessage()
    assert "Warning" in msg
    assert "RuntimeError" in msg
    assert "disk full" in msg


def test_load_initial_glossary_warns_and_returns_empty_when_file_missing(monkeypatch, caplog):
    """H2 fix: when the initial glossary load fails, fall back to empty
    dict but emit a Warning on the logger."""
    from vietdub import translate as translate_mod

    monkeypatch.setattr(
        translate_mod, "load_dictionary_entries",
        lambda _path: (_ for _ in ()).throw(OSError("permission denied")),
    )

    glossary = translate_mod._load_initial_glossary()
    assert glossary == {}
    warnings = [r for r in caplog.records if r.name == "vietdub.translate" and r.levelname == "WARNING"]
    assert warnings, "Expected at least one WARNING log from vietdub.translate"
    msg = warnings[0].getMessage()
    assert "Warning" in msg
    assert "OSError" in msg
    assert "permission denied" in msg


def test_translate_warns_once_for_unknown_model(monkeypatch, caplog):
    """H4 fix: unknown-model warning appears only once per process per model."""
    from vietdub import translate as translate_mod
    from vietdub.translate import translate_with_llm, _warned_unknown_caps
    from vietdub.models import TimedSegment
    import sys, types, json

    # Reset module state for this test.
    monkeypatch.setattr(translate_mod, "_warned_unknown_caps", set())

    unknown_model = "fake-unknown-model-xyz"

    # Patch anthropic.
    class _TextBlock:
        def __init__(self, text): self.text = text; self.type = "text"
    def _make_messages():
        class _Messages:
            def __init__(self): self.calls = []
            def create(self, **kwargs):
                self.calls.append(kwargs)
                user_prompt = kwargs["messages"][0]["content"]
                payload = json.loads(user_prompt)
                segs = payload["segments"]
                # Handle translate pass (id/text) and review pass (segment_id/text_cn).
                def _row(s):
                    if "id" in s:
                        return {
                            "segment_id": s["id"],
                            "start_ms": s["start_ms"],
                            "end_ms": s["end_ms"],
                            "speaker": s.get("speaker"),
                            "text_cn": s["text"],
                            "text_vi": "vi",
                            "context_note": "",
                            "status": "draft",
                        }
                    return {
                        "segment_id": s["segment_id"],
                        "start_ms": s["start_ms"],
                        "end_ms": s["end_ms"],
                        "speaker": s.get("speaker"),
                        "text_cn": s.get("text_cn", ""),
                        "text_vi": s.get("text_vi", "vi"),
                        "context_note": "",
                        "status": "draft",
                    }
                return types.SimpleNamespace(
                    content=[_TextBlock(json.dumps({
                        "translations": [_row(s) for s in segs]
                    }))],
                    stop_reason="end_turn",
                )
        return _Messages()
    class _Anthropic:
        def __init__(self, **kwargs): self.kwargs = kwargs; self.messages = _make_messages()
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="k", anthropic_base_url="https://api.minimax.io/anthropic", llm_model=unknown_model,
    )
    segs = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="hello")]

    # First call: should warn.
    translate_with_llm(segs, {}, settings=settings)
    first_call_warnings = [
        r for r in caplog.records
        if r.name == "vietdub.translate"
        and r.levelname == "WARNING"
        and "not in KNOWN_MODEL_OUTPUT_CAPS" in r.getMessage()
    ]
    warning_count_1 = len(first_call_warnings)

    # Second call: should NOT warn again.
    caplog.clear()
    translate_with_llm(segs, {}, settings=settings)
    second_call_warnings = [
        r for r in caplog.records
        if r.name == "vietdub.translate"
        and r.levelname == "WARNING"
        and "not in KNOWN_MODEL_OUTPUT_CAPS" in r.getMessage()
    ]
    warning_count_2 = len(second_call_warnings)

    assert warning_count_1 == 1, f"Expected 1 warning on first call, got {warning_count_1}"
    assert warning_count_2 == 0, f"Expected 0 warnings on second call, got {warning_count_2}"
    # Direct mechanism check: the model name should be in the warned set
    # after the first call, so a future refactor that uses a different
    # dedup mechanism (e.g., lru_cache) would still need to satisfy this
    # invariant.
    assert unknown_model in translate_mod._warned_unknown_caps


def test_translate_known_model_emits_no_cap_warning(monkeypatch, caplog):
    """H4 sanity: known models (MiniMax-M3) do not emit the unknown-cap warning."""
    from vietdub import translate as translate_mod
    from vietdub.translate import translate_with_llm
    from vietdub.models import TimedSegment
    import sys, types, json

    monkeypatch.setattr(translate_mod, "_warned_unknown_caps", set())

    class _TextBlock:
        def __init__(self, text): self.text = text; self.type = "text"
    def _make_messages():
        class _Messages:
            def __init__(self): self.calls = []
            def create(self, **kwargs):
                user_prompt = kwargs["messages"][0]["content"]
                payload = json.loads(user_prompt)
                segs = payload["segments"]
                # Handle translate pass (id/text) and review pass (segment_id/text_cn).
                def _row(s):
                    if "id" in s:
                        return {
                            "segment_id": s["id"],
                            "start_ms": s["start_ms"],
                            "end_ms": s["end_ms"],
                            "speaker": s.get("speaker"),
                            "text_cn": s["text"],
                            "text_vi": "vi",
                            "context_note": "",
                            "status": "draft",
                        }
                    return {
                        "segment_id": s["segment_id"],
                        "start_ms": s["start_ms"],
                        "end_ms": s["end_ms"],
                        "speaker": s.get("speaker"),
                        "text_cn": s.get("text_cn", ""),
                        "text_vi": s.get("text_vi", "vi"),
                        "context_note": "",
                        "status": "draft",
                    }
                return types.SimpleNamespace(
                    content=[_TextBlock(json.dumps({
                        "translations": [_row(s) for s in segs]
                    }))],
                    stop_reason="end_turn",
                )
        return _Messages()
    class _Anthropic:
        def __init__(self, **kwargs): self.kwargs = kwargs; self.messages = _make_messages()
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

    settings = types.SimpleNamespace(
        anthropic_api_key="k", anthropic_base_url="https://api.minimax.io/anthropic", llm_model="MiniMax-M3",
    )
    segs = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="hello")]
    translate_with_llm(segs, {}, settings=settings)
    cap_warnings = [
        r for r in caplog.records
        if r.name == "vietdub.translate"
        and r.levelname == "WARNING"
        and "not in KNOWN_MODEL_OUTPUT_CAPS" in r.getMessage()
    ]
    assert not cap_warnings, f"Expected no cap warning for known model, got: {[w.getMessage() for w in cap_warnings]}"


