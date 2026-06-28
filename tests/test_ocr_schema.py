import pytest
from vietdub.ocr.schema import OcrSegment, OcrSegmentError


def test_valid_segment():
    s = OcrSegment(id="m-0001", start_ms=0, end_ms=2440, text="hello", confidence=0.95)
    assert s.start_ms == 0
    assert s.end_ms == 2440
    assert s.confidence == 0.95


def test_confidence_optional():
    s = OcrSegment(id="x", start_ms=0, end_ms=100, text="hi")
    assert s.confidence is None


def test_confidence_out_of_range():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=0, end_ms=100, text="hi", confidence=1.5)


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=200, end_ms=100, text="hi")


def test_negative_start_rejected():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=-1, end_ms=100, text="hi")