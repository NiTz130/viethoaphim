import json
import sys
import types

import pytest

from vietdub.models import TimedSegment
from vietdub.translate import translate_with_llm


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class _FakeMessages:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            content=[_TextBlock(json.dumps({
                "translations": [{
                    "segment_id": "m-0001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker": None,
                    "text_cn": "你好",
                    "text_vi": "Xin chào",
                    "context_note": "",
                    "status": "translated",
                }]
            }))]
        )


class _FakeAnthropic:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = _FakeMessages()
        self.__class__.instances.append(self)


def _settings(**overrides):
    base = {
        "anthropic_api_key": "test-key",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
        "llm_model": "MiniMax-M3",
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_translate_with_llm_calls_anthropic_messages_endpoint(monkeypatch):
    fake_anthropic = types.SimpleNamespace(Anthropic=_FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
        context_bundle={},
        settings=_settings(),
    )

    client = _FakeAnthropic.instances[-1]
    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["base_url"] == "https://api.minimax.io/anthropic"
    call = client.messages.calls[0]
    assert call["model"] == "MiniMax-M3"
    assert "system" in call
    assert isinstance(call["messages"], list)
    assert rows[0].text_vi == "Xin chào"


def test_translate_with_llm_maps_authentication_error(monkeypatch):
    class _AuthError(Exception):
        def __init__(self):
            super().__init__("bad key")

    class _BoomMessages:
        def create(self, **kwargs):
            raise _AuthError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        AuthenticationError=_AuthError,
    ))

    with pytest.raises(RuntimeError, match="authentication failed"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_maps_http_status_error(monkeypatch):
    class _StatusError(Exception):
        status_code = 429
        message = "rate limited"

    class _BoomMessages:
        def create(self, **kwargs):
            raise _StatusError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        APIStatusError=_StatusError,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    with pytest.raises(RuntimeError, match="HTTP 429"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_maps_connection_error(monkeypatch):
    class _ConnError(Exception):
        pass

    class _BoomMessages:
        def create(self, **kwargs):
            raise _ConnError()

    class _BoomAnthropic:
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_BoomAnthropic,
        APIConnectionError=_ConnError,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
    ))

    with pytest.raises(RuntimeError, match="network error"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_rejects_empty_content(monkeypatch):
    class _EmptyMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(content=[])

    class _EmptyAnthropic:
        def __init__(self, **kwargs):
            self.messages = _EmptyMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_EmptyAnthropic))

    with pytest.raises(RuntimeError, match="empty response"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_requires_api_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(anthropic_api_key=""),
        )