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
        APIStatusError=_AuthError,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    with pytest.raises(RuntimeError, match="authentication failed"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_retries_then_gives_up_on_rate_limit(monkeypatch):
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
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="4 attempts"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_retries_then_gives_up_on_connection_error(monkeypatch):
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
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="4 attempts"):
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )


def test_translate_with_llm_does_not_retry_non_retryable_http_status(monkeypatch):
    class _StatusError(Exception):
        status_code = 400
        message = "bad request"

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

    with pytest.raises(RuntimeError, match="HTTP 400"):
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


def test_translate_one_batch_with_retry_recovers_from_transient_connection_error(monkeypatch):
    class _ConnError(Exception):
        pass

    call_count = [0]

    class _CountingMessages:
        def create(self, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise _ConnError()
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
                        "status": "draft",
                    }]
                }))],
                stop_reason="end_turn",
            )

    class _CountingAnthropic:
        def __init__(self, **kwargs):
            self.messages = _CountingMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_CountingAnthropic,
        APIConnectionError=_ConnError,
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    ))

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    from vietdub.translate import _translate_one_batch_with_retry
    rows = _translate_one_batch_with_retry(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好")],
        context_bundle={},
        settings=_settings(),
    )

    assert call_count[0] == 3
    assert sleep_calls == [1.0, 2.0]
    assert rows[0].text_vi == "Xin chào"


def test_translate_one_batch_with_retry_gives_up_after_max_attempts(monkeypatch):
    class _ConnError(Exception):
        pass

    class _AlwaysFailMessages:
        def create(self, **kwargs):
            raise _ConnError()

    class _AlwaysFailAnthropic:
        def __init__(self, **kwargs):
            self.messages = _AlwaysFailMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_AlwaysFailAnthropic,
        APIConnectionError=_ConnError,
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    ))

    monkeypatch.setattr("time.sleep", lambda _: None)

    from vietdub.translate import _translate_one_batch_with_retry
    with pytest.raises(RuntimeError, match="4 attempts") as exc_info:
        _translate_one_batch_with_retry(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, _ConnError)


def test_translate_one_batch_with_retry_does_not_retry_non_retryable_status(monkeypatch):
    class _AuthError(Exception):
        def __str__(self):
            return "bad key"

    class _AuthMessages:
        def create(self, **kwargs):
            raise _AuthError()

    class _AuthAnthropic:
        def __init__(self, **kwargs):
            self.messages = _AuthMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_AuthAnthropic,
        AuthenticationError=_AuthError,
        APIStatusError=_AuthError,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    from vietdub.translate import _translate_one_batch_with_retry
    with pytest.raises(RuntimeError, match="authentication failed"):
        _translate_one_batch_with_retry(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(),
        )

    # Call count check: raise from messages.create is caught once, wrapped, re-raised.
    # Verify no retries happened.
    assert sleep_calls == []


def test_translate_one_batch_detects_max_tokens_truncation(monkeypatch):
    class _TruncatedMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [
                        {
                            "segment_id": "m-0001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "speaker": None,
                            "text_cn": "你好",
                            "text_vi": "Xin chào",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0002",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "speaker": None,
                            "text_cn": "世界",
                            "text_vi": "Thế giới",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0003",
                            "start_ms": 2000,
                            "end_ms": 3000,
                            "speaker": None,
                            "text_cn": "朋友",
                            "text_vi": "Bạn bè",
                            "context_note": "",
                            "status": "draft",
                        },
                    ]
                }))],
                stop_reason="max_tokens",
            )

    class _TruncatedAnthropic:
        def __init__(self, **kwargs):
            self.messages = _TruncatedMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_TruncatedAnthropic,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    from vietdub.translate import _translate_one_batch
    with pytest.raises(RuntimeError, match="max_tokens") as exc_info:
        _translate_one_batch(
            segments=[
                TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好"),
                TimedSegment(id="m-0002", start_ms=1000, end_ms=2000, text="世界"),
                TimedSegment(id="m-0003", start_ms=2000, end_ms=3000, text="朋友"),
            ],
            context_bundle={},
            settings=_settings(),
        )

    msg = str(exc_info.value)
    assert "3" in msg
    assert "Reduce LLM_BATCH_SIZE" in msg


def test_translate_one_batch_detects_row_count_mismatch(monkeypatch):
    class _ShortMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_TextBlock(json.dumps({
                    "translations": [
                        {
                            "segment_id": "m-0001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "speaker": None,
                            "text_cn": "你好",
                            "text_vi": "Xin chào",
                            "context_note": "",
                            "status": "draft",
                        },
                        {
                            "segment_id": "m-0002",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "speaker": None,
                            "text_cn": "世界",
                            "text_vi": "Thế giới",
                            "context_note": "",
                            "status": "draft",
                        },
                    ]
                }))],
                stop_reason="end_turn",
            )

    class _ShortAnthropic:
        def __init__(self, **kwargs):
            self.messages = _ShortMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=_ShortAnthropic,
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {"status_code": 500, "message": ""}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
    ))

    from vietdub.translate import _translate_one_batch
    with pytest.raises(RuntimeError, match="count mismatch") as exc_info:
        _translate_one_batch(
            segments=[
                TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好"),
                TimedSegment(id="m-0002", start_ms=1000, end_ms=2000, text="世界"),
                TimedSegment(id="m-0003", start_ms=2000, end_ms=3000, text="朋友"),
            ],
            context_bundle={},
            settings=_settings(),
        )

    msg = str(exc_info.value)
    assert "2" in msg
    assert "3" in msg


