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
