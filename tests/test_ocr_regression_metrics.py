import pytest
from vietdub.ocr.schema import OcrSegment
from vietdub.ocr.regression import match_segments, compute_metrics


@pytest.fixture
def baseline():
    return [
        OcrSegment(id="a", start_ms=0, end_ms=1000, text="hello world"),
        OcrSegment(id="b", start_ms=1000, end_ms=2000, text="foo bar"),
        OcrSegment(id="c", start_ms=2000, end_ms=3000, text="baz qux"),
    ]


@pytest.fixture
def new_close(baseline):
    return [
        OcrSegment(id="x", start_ms=10, end_ms=990, text="hello world"),
        OcrSegment(id="y", start_ms=1000, end_ms=2000, text="foo bar"),
        OcrSegment(id="z", start_ms=2000, end_ms=3000, text="baz qux"),
    ]


def test_match_pairs_by_midpoint(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    assert ("a", "x") in pairs
    assert ("b", "y") in pairs
    assert ("c", "z") in pairs


def test_match_threshold_excludes_far_segments():
    base = [OcrSegment(id="a", start_ms=0, end_ms=1000, text="hi")]
    new = [OcrSegment(id="z", start_ms=10000, end_ms=11000, text="hi")]
    pairs = match_segments(base, new, threshold_ms=200)
    assert pairs == []


def test_match_excludes_zero_duration():
    base = [OcrSegment(id="a", start_ms=0, end_ms=0, text="x")]
    new = [OcrSegment(id="z", start_ms=0, end_ms=1000, text="x")]
    pairs = match_segments(base, new)
    assert pairs == []


def test_metrics_segment_count(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    assert m["segment_count"]["baseline"] == 3
    assert m["segment_count"]["new"] == 3
    assert m["segment_count"]["delta_pct"] == 0.0


def test_metrics_text_similarity_perfect(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    assert m["text_similarity"] == 1.0


def test_metrics_iou_perfect(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    # first pair: intersection 980, union 1000 -> 0.98
    assert 0.95 <= m["time_range_overlap"] <= 1.0


def test_metrics_unmatched_pct():
    base = [OcrSegment(id=f"b{i}", start_ms=i*1000, end_ms=i*1000+500, text="x") for i in range(10)]
    new = [OcrSegment(id=f"n{i}", start_ms=i*1000, end_ms=i*1000+500, text="x") for i in range(8)]
    pairs = match_segments(base, new)
    m = compute_metrics(base, new, pairs)
    # 2 baseline unmatched, 8 pairs -> unmatched_pct = 2/10 * 100 = 20.0
    assert m["unmatched_pct"] == 20.0
