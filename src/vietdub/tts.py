from __future__ import annotations

from pathlib import Path

from .models import TranslationRow


class EdgeTtsEngine:
    def __init__(self, voice: str) -> None:
        self.voice = voice

    async def synthesize_segment(self, row: TranslationRow, output: Path) -> Path:
        import edge_tts

        output.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(row.text_vi, self.voice)
        await communicate.save(str(output))
        return output
