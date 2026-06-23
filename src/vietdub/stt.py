from __future__ import annotations

import json
from pathlib import Path

from .models import TimedSegment


class FixtureSttEngine:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path

    def transcribe(self, audio_path: Path) -> list[TimedSegment]:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return [TimedSegment.model_validate(item) for item in data]


class FasterWhisperSttEngine:
    def __init__(self, model_name: str, language: str = "zh") -> None:
        self.model_name = model_name
        self.language = language

    def transcribe(self, audio_path: Path) -> list[TimedSegment]:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, device="auto", compute_type="auto")
        segments, _info = model.transcribe(str(audio_path), language=self.language)
        result: list[TimedSegment] = []
        for index, segment in enumerate(segments, start=1):
            result.append(
                TimedSegment(
                    id=f"stt-{index:04}",
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                    text=segment.text.strip(),
                    confidence=None,
                    source="stt",
                )
            )
        return result
