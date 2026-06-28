"""OCR package."""
from .paddle import (
    FixtureOcrEngine,
    PaddleSubtitleOcrEngine,
    _clear_generated_ocr_frames,
    _guard_optional_torch_import,
)

from .schema import OcrSegment, OcrSegmentError

__all__ = [
    "FixtureOcrEngine",
    "PaddleSubtitleOcrEngine",
    "OcrSegment",
    "OcrSegmentError",
]
