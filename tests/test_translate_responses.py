import json
import sys
import types

from vietdub.models import TimedSegment
from vietdub.translate import translate_with_llm


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            output_text=json.dumps(
                {
                    "translations": [
                        {
                            "segment_id": "m-0001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "speaker": None,
                            "text_cn": "\u4f60\u597d",
                            "text_vi": "Xin ch\u00e0o",
                            "context_note": "",
                            "status": "draft",
                        }
                    ]
                }
            )
        )


class FakeChatCompletions:
    def create(self, **kwargs):
        raise AssertionError("chat completions should not be used for /responses endpoint")


class FakeOpenAI:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.responses = FakeResponses()
        self.chat = types.SimpleNamespace(completions=FakeChatCompletions())
        self.__class__.instances.append(self)


class FakeAuthenticationError(Exception):
    pass


class FakeAPIStatusError(Exception):
    status_code = 500


class FakeOpenAIError(Exception):
    pass


def test_translate_with_llm_uses_responses_api_when_base_url_is_responses_endpoint(monkeypatch):
    fake_openai = types.SimpleNamespace(
        APIStatusError=FakeAPIStatusError,
        AuthenticationError=FakeAuthenticationError,
        OpenAI=FakeOpenAI,
        OpenAIError=FakeOpenAIError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    rows = translate_with_llm(
        segments=[TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d")],
        context_bundle={},
        api_key="test-key",
        model="test-model",
        base_url="https://luongchidung.online/v1/responses",
    )

    client = FakeOpenAI.instances[-1]
    assert client.kwargs["base_url"] == "https://luongchidung.online/v1"
    assert client.responses.calls[0]["model"] == "test-model"
    assert "Translate Chinese cartoon dialogue" in client.responses.calls[0]["input"]
    assert rows[0].text_vi == "Xin ch\u00e0o"
