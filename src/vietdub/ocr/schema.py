"""OCR segment schema with pydantic validation."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class OcrSegmentError(Exception):
    """Raised when an OCR segment fails schema validation."""


class OcrSegment(BaseModel):
    """A single OCR'd subtitle segment from PaddleOCR output.

    Matches the JSON shape written to jobs/<name>/ocr/subtitles.json.
    """

    id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_duration(self) -> "OcrSegment":
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            )
        return self
