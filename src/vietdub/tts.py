from __future__ import annotations

from pathlib import Path

import httpx

from .models import TranslationRow


class MiniMaxTtsEngine:
    def __init__(
        self,
        voice_id: str,
        api_key: str,
        base_url: str,
        model: str = "speech-2.8-hd",
        timeout: float = 30.0,
    ) -> None:
        self.voice_id = voice_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def synthesize_segment(self, row: TranslationRow, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "text": row.text_vi,
            "voice_setting": {"voice_id": self.voice_id, "language_boost": "auto"},
            "audio_setting": {"format": "mp3", "sample_rate": 24000},
            "output_format": "hex",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/t2a_v2"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"MiniMax TTS HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"MiniMax TTS network error: {exc}") from exc

        data = resp.json()
        base = data.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            raise RuntimeError(
                f"MiniMax TTS error: {base.get('status_msg', 'unknown')} "
                f"(code={base.get('status_code')})"
            )
        audio_hex = data.get("data", {}).get("audio", "")
        if not audio_hex:
            raise RuntimeError("MiniMax TTS returned empty audio")
        output.write_bytes(bytes.fromhex(audio_hex))
        return output
