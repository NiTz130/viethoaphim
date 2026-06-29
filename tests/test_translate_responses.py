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
            }))],
            stop_reason="end_turn",
        )


class _FakeAnthropic:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = _FakeMessages()
        self.__class__.instances.append(self)


class _BatchAwareMessages:
    """Mock that returns rows matching the input batch size and segment IDs."""

    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Parse segment count from user_prompt (JSON-encoded in messages[0]["content"])
        user_prompt = kwargs["messages"][0]["content"]
        payload = json.loads(user_prompt)
        batch_segments = payload["segments"]
        # Translate prompts use {"id": ...} for segments; review prompts use {"segment_id": ...}
        translations = [
            {
                "segment_id": seg.get("id") or seg["segment_id"],
                "start_ms": seg.get("start_ms", 0),
                "end_ms": seg.get("end_ms", 1000),
                "speaker": seg.get("speaker"),
                "text_cn": seg.get("text", ""),
                "text_vi": f"translation of {seg.get('id') or seg['segment_id']}",
                "context_note": "",
                "status": "draft",
            }
            for seg in batch_segments
        ]
        return types.SimpleNamespace(
            content=[_TextBlock(json.dumps({"translations": translations}))],
            stop_reason="end_turn",
        )


class _BatchAwareAnthropic:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = _BatchAwareMessages()
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

    assert call_count[0] == 4  # 2 failed translate + 1 successful translate + 1 review
    assert sleep_calls == [1.0, 2.0]
    assert rows[0].text_vi == "Xin chào"


def test_translate_one_batch_with_retry_gives_up_after_max_attempts(monkeypatch):
    class _ConnError(Exception):
        pass

    call_count = [0]

    class _AlwaysFailMessages:
        def create(self, **kwargs):
            call_count[0] += 1
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
    assert call_count[0] == 4


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


def test_translate_with_llm_raises_when_llm_max_tokens_exceeds_model_cap(monkeypatch):
    """M3: the cap-exceeded check still fires when max_tokens > cap for any
    in-dict model. We patch LLM_MAX_TOKENS to a value larger than every
    cap in KNOWN_MODEL_OUTPUT_CAPS so the test is robust to dict refreshes."""
    from vietdub import translate as translate_mod
    monkeypatch.setattr(translate_mod, "LLM_MAX_TOKENS", 16384)
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    with pytest.raises(RuntimeError, match="exceeds known output cap") as exc_info:
        translate_with_llm(
            segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
            context_bundle={},
            settings=_settings(llm_model="claude-haiku-4-5-20251001"),
        )

    msg = str(exc_info.value)
    assert "16384" in msg
    assert "8192" in msg


def test_translate_with_llm_passes_when_model_cap_sufficient(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
        context_bundle={},
        settings=_settings(llm_model="claude-3-5-sonnet-20240620"),
    )

    assert len(rows) == 1


def test_translate_with_llm_warns_for_unknown_model(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_FakeAnthropic))

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="x")],
        context_bundle={},
        settings=_settings(llm_model="some-future-model-xyz"),
    )

    assert len(rows) == 1
    captured = capsys.readouterr()
    assert "some-future-model-xyz" in captured.err
    assert "KNOWN_MODEL_OUTPUT_CAPS" in captured.err


def test_translate_with_llm_batches_50_segments_into_one_call(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(50)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 50
    # With review=True (default), 1 batch × 2 calls = 2 Anthropic instances (translate + review)
    assert len(_BatchAwareAnthropic.instances) == 2
    # Each instance handles one call (translate client, then review client)
    assert len(_BatchAwareAnthropic.instances[0].messages.calls) == 1
    assert len(_BatchAwareAnthropic.instances[1].messages.calls) == 1


def test_translate_with_llm_batches_51_segments_into_two_calls(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(51)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 51
    # With review=True (default), 2 batches × 2 calls (translate + review) = 4 instances
    assert len(_BatchAwareAnthropic.instances) == 4
    # First call covers m-0000..m-0049 (50 segments)
    first_payload = json.loads(
        _BatchAwareAnthropic.instances[0].messages.calls[0]["messages"][0]["content"]
    )
    assert len(first_payload["segments"]) == 50
    assert first_payload["segments"][0]["id"] == "m-0000"
    # Third call (index 2) covers m-0050 (1 segment) — second batch's translate call
    second_payload = json.loads(
        _BatchAwareAnthropic.instances[2].messages.calls[0]["messages"][0]["content"]
    )
    assert len(second_payload["segments"]) == 1
    assert second_payload["segments"][0]["id"] == "m-0050"


def test_translate_with_llm_batches_100_segments_into_two_calls(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    segments = [
        TimedSegment(id=f"m-{i:04d}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"text {i}")
        for i in range(100)
    ]

    rows = translate_with_llm(segments=segments, context_bundle={}, settings=_settings())

    assert len(rows) == 100
    # With review=True (default), 2 batches × 2 calls = 4 instances
    assert len(_BatchAwareAnthropic.instances) == 4


def test_translate_with_llm_handles_empty_segments(monkeypatch):
    _BatchAwareAnthropic.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_BatchAwareAnthropic))

    rows = translate_with_llm(segments=[], context_bundle={}, settings=_settings())

    assert rows == []
    assert len(_BatchAwareAnthropic.instances) == 0


