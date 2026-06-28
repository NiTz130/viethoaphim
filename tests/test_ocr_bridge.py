from pathlib import Path

import pytest

from vietdub.ocr.bridge import ocr_to_segments, run_pipeline_ocr


class _FakeEngine:
    def __init__(self) -> None:
        self.received_path = None

    def recognize(self, video_path: Path):
        # Record the type for assertion below.
        self.received_path = video_path
        from vietdub.models import TimedSegment

        return [
            TimedSegment(
                id="ocr-0001",
                start_ms=0,
                end_ms=500,
                text="hi",
                source="ocr",
            )
        ]


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
    from vietdub.ocr.schema import OcrSegmentError
    with pytest.raises(OcrSegmentError):
        ocr_to_segments([{"start_ms": 0, "end_ms": 100, "text": "x"}])  # no id


def test_run_pipeline_ocr_converts_timed_segments(monkeypatch):
    fake = _FakeEngine()
    # Patch the symbol at the location the bridge imports it from
    # (lazy import inside run_pipeline_ocr: `from .paddle import PaddleSubtitleOcrEngine`).
    monkeypatch.setattr(
        "vietdub.ocr.paddle.PaddleSubtitleOcrEngine",
        lambda: fake,
    )

    result = run_pipeline_ocr(Path("dummy.mp4"))

    # Engine received a Path object, not a str.
    assert isinstance(fake.received_path, Path)

    # And the bridge produced an OcrSegment with only OcrSegment fields.
    assert len(result) == 1
    seg = result[0]
    assert seg.id == "ocr-0001"
    assert seg.start_ms == 0
    assert seg.end_ms == 500
    assert seg.text == "hi"
    assert seg.confidence is None
    # No extra TimedSegment fields leaked into the OcrSegment.
    assert not hasattr(seg, "source")
    assert not hasattr(seg, "speaker")
    assert not hasattr(seg, "meta")