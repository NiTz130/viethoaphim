from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_TONE = "h\u00e0i sa \u0111i\u00eau, t\u1ef1 nhi\u00ean ti\u1ebfng Vi\u1ec7t"


class StepName(StrEnum):
    EXTRACT = "extract"
    STT = "stt"
    OCR = "ocr"
    MERGE = "merge"
    CONTEXT = "context"
    TRANSLATE = "translate"
    TTS = "tts"
    RENDER = "render"


STEP_ORDER: list[StepName] = [
    StepName.EXTRACT,
    StepName.STT,
    StepName.OCR,
    StepName.MERGE,
    StepName.CONTEXT,
    StepName.TRANSLATE,
    StepName.TTS,
    StepName.RENDER,
]


class TimedSegment(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    confidence: float | None = None
    source: str = "unknown"
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timing(self) -> "TimedSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class TranslationRow(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    speaker: str | None = None
    text_cn: str
    text_vi: str
    context_note: str = ""
    status: str = "draft"

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"draft", "reviewed", "skip"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value


class SyncSegmentReport(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    original_duration_ms: int
    synced_duration_ms: int
    speed_factor: float
    warnings: list[str] = Field(default_factory=list)


class SyncReport(BaseModel):
    output_audio: Path
    sample_rate: int = 44_100
    segments: list[SyncSegmentReport] = Field(default_factory=list)


def millis_to_srt_time(ms: int) -> str:
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
