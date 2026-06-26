import json as json_module
import sys
import types

import pytest

from vietdub.models import TranslationRow
from vietdub.tts import MiniMaxTtsEngine


def _row(text: str = "Xin chào", segment_id: str = "m-0001") -> TranslationRow:
    return TranslationRow(
        segment_id=segment_id,
        start_ms=0,
        end_ms=1000,
        speaker=None,
        text_cn="source",
        text_vi=text,
        context_note="",
        status="reviewed",
    )


class _FakeAsyncClient:
    def __init__(self, response_payload: dict, **kwargs) -> None:
        self.kwargs = kwargs
        self.response_payload = response_payload
        self.post_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json=None, headers=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        return types.SimpleNamespace(
            status_code=200,
            text=json_module.dumps(self.response_payload),
            json=lambda: self.response_payload,
            raise_for_status=lambda: None,
        )


def _install_fake_httpx(monkeypatch, response_payload: dict, **client_kwargs):
    fake = _FakeAsyncClient(response_payload, **client_kwargs)

    class _FakeAsyncClientClass:
        def __init__(self, **kwargs):
            fake.kwargs = kwargs
            self._fake = fake

        async def __aenter__(self):
            return self._fake

        async def __aexit__(self, *args):
            return None

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClientClass)
    return fake


@pytest.mark.asyncio
async def test_minimax_tts_engine_writes_hex_decoded_audio(tmp_path, monkeypatch):
    payload_bytes = b"FAKE_MP3_CONTENT"
    response_payload = {
        "data": {"audio": payload_bytes.hex(), "status": 2},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    fake = _install_fake_httpx(monkeypatch, response_payload)

    engine = MiniMaxTtsEngine(
        voice_id="vi-female-1",
        api_key="test-key",
        base_url="https://api.minimax.io/v1",
    )
    output = tmp_path / "segment.mp3"
    result = await engine.synthesize_segment(_row(), output)

    assert result == output
    assert output.read_bytes() == payload_bytes
    assert fake.post_calls[0]["url"] == "https://api.minimax.io/v1/t2a_v2"
    assert fake.post_calls[0]["headers"]["Authorization"] == "Bearer test-key"
    sent = fake.post_calls[0]["json"]
    assert sent["text"] == "Xin chào"
    assert sent["voice_setting"]["voice_id"] == "vi-female-1"
    assert sent["voice_setting"]["language_boost"] == "auto"
    assert sent["model"] == "speech-2.8-hd"
    assert sent["output_format"] == "hex"


@pytest.mark.asyncio
async def test_minimax_tts_engine_raises_on_http_error(tmp_path, monkeypatch):
    class _BoomClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None, headers=None):
            class _Resp:
                status_code = 401
                text = "unauthorized"
                def raise_for_status(self):
                    import httpx
                    raise httpx.HTTPStatusError("401", request=None, response=self)
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)

    engine = MiniMaxTtsEngine(voice_id="v", api_key="bad", base_url="https://x")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        await engine.synthesize_segment(_row(), tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_minimax_tts_engine_raises_on_baseresp_error(tmp_path, monkeypatch):
    payload = {
        "data": {"audio": "", "status": 1},
        "base_resp": {"status_code": 1001, "status_msg": "voice not found"},
    }
    _install_fake_httpx(monkeypatch, payload)

    engine = MiniMaxTtsEngine(voice_id="bad", api_key="k", base_url="https://x")
    with pytest.raises(RuntimeError, match="voice not found"):
        await engine.synthesize_segment(_row(), tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_minimax_tts_engine_raises_on_network_error(tmp_path, monkeypatch):
    class _BoomClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None, headers=None):
            import httpx
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.AsyncClient", _BoomClient)

    engine = MiniMaxTtsEngine(voice_id="v", api_key="k", base_url="https://x")
    with pytest.raises(RuntimeError, match="network error"):
        await engine.synthesize_segment(_row(), tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_minimax_tts_engine_raises_when_api_key_missing(tmp_path):
    engine = MiniMaxTtsEngine(voice_id="v", api_key="", base_url="https://x")
    with pytest.raises(RuntimeError, match="TTS_API_KEY is required"):
        await engine.synthesize_segment(_row(), tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_minimax_tts_engine_appends_t2a_v2_to_base_url(tmp_path, monkeypatch):
    """Lock in the contract: base_url is a base, engine appends /t2a_v2.
    Catches the regression where the default tts_base_url includes /t2a_v2
    and the engine double-appends it, producing a 404."""
    payload = {
        "data": {"audio": b"ok".hex(), "status": 2},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    fake = _install_fake_httpx(monkeypatch, payload)

    engine = MiniMaxTtsEngine(
        voice_id="v",
        api_key="k",
        base_url="https://api.minimax.io/v1",
    )
    await engine.synthesize_segment(_row(), tmp_path / "out.mp3")

    assert fake.post_calls[0]["url"] == "https://api.minimax.io/v1/t2a_v2"
