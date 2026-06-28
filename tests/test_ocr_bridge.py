from vietdub.ocr.bridge import ocr_to_segments


def test_ocr_to_segments_coerces_dict():
    raw = [
        {"id": "m-0001", "start_ms": 0, "end_ms": 1000, "text": "hi", "confidence": 0.9},
        {"id": "m-0002", "start_ms": 1000, "end_ms": 2000, "text": "bye"},
    ]
    segs = ocr_to_segments(raw)
    assert len(segs) == 2
    assert segs[0].text == "hi"
    assert segs[1].confidence is None


def test_ocr_to_segments_rejects_missing_required():
    import pytest
    from vietdub.ocr.schema import OcrSegmentError
    with pytest.raises(OcrSegmentError):
        ocr_to_segments([{"start_ms": 0, "end_ms": 100, "text": "x"}])  # no id
