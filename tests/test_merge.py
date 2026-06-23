from vietdub.merge import merge_segments
from vietdub.models import TimedSegment


def test_ocr_text_wins_when_timing_overlaps():
    stt = [
        TimedSegment(
            id="s1",
            start_ms=0,
            end_ms=1800,
            text="\u6000\u7591\u4f60\u6709\u795e\u79d8\u8fc7\u53bb",
            source="stt",
        )
    ]
    ocr = [
        TimedSegment(
            id="o1",
            start_ms=100,
            end_ms=1700,
            text="\u90fd\u6000\u7591\u4f60\u6709\u4e00\u4e2a\u795e\u79d8\u7684\u8fc7\u5f80",
            source="ocr",
        )
    ]

    merged = merge_segments(stt, ocr)

    assert len(merged) == 1
    assert merged[0].text == "\u90fd\u6000\u7591\u4f60\u6709\u4e00\u4e2a\u795e\u79d8\u7684\u8fc7\u5f80"
    assert merged[0].source == "ocr+stt"


def test_non_overlapping_stt_is_kept():
    stt = [TimedSegment(id="s1", start_ms=3000, end_ms=4000, text="\u4f60\u597d", source="stt")]
    ocr = [TimedSegment(id="o1", start_ms=0, end_ms=1000, text="\u5b57\u5e55", source="ocr")]

    merged = merge_segments(stt, ocr)

    assert [segment.text for segment in merged] == ["\u5b57\u5e55", "\u4f60\u597d"]
