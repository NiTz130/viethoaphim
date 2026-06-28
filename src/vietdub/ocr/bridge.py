"""Bridge between vietdub's PaddleOCR pipeline and the regression schema."""
from pathlib import Path
from typing import Iterable

from ..models import TimedSegment
from .schema import OcrSegment, OcrSegmentError


def ocr_to_segments(raw: Iterable[dict]) -> list[OcrSegment]:
    """Convert raw OCR dicts (as written by the pipeline) to OcrSegment models.

    Raises OcrSegmentError if any segment is malformed (does not silently skip).
    """
    out: list[OcrSegment] = []
    for i, item in enumerate(raw):
        try:
            out.append(OcrSegment.model_validate(item))
        except Exception as e:
            raise OcrSegmentError(f"segment index {i} malformed: {e}") from e
    return out


def _timed_segment_to_ocr_dict(seg: TimedSegment) -> dict:
    """Convert a TimedSegment into a dict with only the keys OcrSegment accepts.

    Drops extra fields like ``source``, ``meta``, ``speaker``.
    """
    return {
        "id": seg.id,
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "text": seg.text,
    }


def run_pipeline_ocr(video_path, region: str = "bottom_28pct") -> list[OcrSegment]:
    """Run the vietdub PaddleOCR engine on a video file.

    Imports lazily so unit tests don't require paddlepaddle.
    """
    from .paddle import PaddleSubtitleOcrEngine
    engine = PaddleSubtitleOcrEngine()
    raw_segments: list[TimedSegment] = engine.recognize(Path(video_path))
    raw_dicts = [_timed_segment_to_ocr_dict(seg) for seg in raw_segments]
    return ocr_to_segments(raw_dicts)