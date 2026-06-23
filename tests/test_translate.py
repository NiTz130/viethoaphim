from vietdub.models import TimedSegment, TranslationRow
from vietdub.translate import export_review_csv, import_review_csv


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
