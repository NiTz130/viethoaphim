import json

import pytest

from vietdub.models import TimedSegment, TranslationRow
from vietdub.translate import (
    build_translation_prompt,
    export_review_csv,
    import_review_csv,
    parse_translation_response,
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


def test_parse_translation_response_warns_and_coerces_invalid_status(capsys):
    """H4 fix: LLM returns an invalid status; we log to stderr AND coerce to 'draft'."""
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
    captured = capsys.readouterr()
    assert "invalid status" in captured.err
    assert "m-0001" in captured.err


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

    from vietdub.translate import translate_with_llm
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
