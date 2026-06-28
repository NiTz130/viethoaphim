"""Bridge between vietdub's PaddleOCR pipeline and the regression schema."""
from typing import Iterable

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


def run_pipeline_ocr(video_path, region: str = "bottom_28pct") -> list[OcrSegment]:
    """Run the vietdub PaddleOCR engine on a video file.

    Imports lazily so unit tests don't require paddlepaddle.
    """
    from .paddle import PaddleSubtitleOcrEngine
    engine = PaddleSubtitleOcrEngine()
    raw = engine.recognize(str(video_path))
    return ocr_to_segments(raw)
