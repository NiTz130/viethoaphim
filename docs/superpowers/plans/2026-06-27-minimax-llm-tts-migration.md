# MiniMax LLM + TTS Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OpenAI SDK (LLM) and edge_tts (TTS) with MiniMax APIs using `anthropic` SDK and a new `MiniMaxTtsEngine` (httpx-based), preserving all public function signatures.

**Architecture:** Hard rename of `openai_*` settings to `anthropic_*`; new `tts_*` settings for MiniMax Speech 2.8; new `MiniMaxTtsEngine` class replaces `EdgeTtsEngine`; one new test file (`tests/test_tts.py`); one test file rewrite (`tests/test_translate_responses.py`).

**Tech Stack:** Python 3.11, anthropic SDK, httpx, pydantic, pytest

## Global Constraints

- Run tests with: `python -m pytest 2>&1 | tail -20`
- LLM endpoint: `https://api.minimax.io/anthropic`
- LLM default model: `MiniMax-M3`
- TTS endpoint: `https://api.minimax.io/v1/t2a_v2`
- TTS default model: `speech-2.8-hd`
- TTS default voice_id: `vi-VN-HoaiMyNeural` (placeholder; almost certainly invalid for MiniMax — user must override)
- TTS per-request timeout: 30s
- Hard rename: no `OPENAI_API_KEY` alias, no OpenAI SDK fallback
- OCR (PaddleOCR) and STT (Faster-Whisper) unchanged
- H4 fix: log warning + accept as `"draft"` on invalid status (do not reject)
- Commit messages use imperative mood, scoped prefix (`fix:`, `test:`, `feat:`, `docs:`, `chore:`)
- Each task ends with a clean commit; full suite must be green after each task

---

### Task 1: Update config + dependencies

**Files:**
- Modify: `src/vietdub/config.py`
- Modify: `pyproject.toml`
- Test: existing `tests/test_models_config.py` (no new test)

**Interfaces:**
- Consumes: existing pydantic-settings flow
- Produces: new `Settings` field names; `anthropic` and `httpx` in deps; `openai` and `edge-tts` removed

- [ ] **Step 1: Update `src/vietdub/config.py`**

Replace the entire file with:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.minimax.io/anthropic"
    llm_model: str = "MiniMax-M3"

    tts_voice_id: str = "vi-VN-HoaiMyNeural"
    tts_model: str = "speech-2.8-hd"
    tts_base_url: str = "https://api.minimax.io/v1/t2a_v2"
    tts_api_key: str = ""
    tts_timeout: float = 30.0

    stt_model: str = "medium"
    stt_language: str = "zh"
    jobs_dir: str = "jobs"
    reference_data_dir: str = "data"
    sample_rate: int = 44_100
```

- [ ] **Step 2: Update `pyproject.toml` dependencies**

Replace the `dependencies` list with:

```toml
  "typer>=0.12.0",
  "pydantic>=2.7.0",
  "pydantic-settings>=2.2.0",
  "python-dotenv>=1.0.1",
  "anthropic>=0.40.0",
  "httpx>=0.27.0",
  "pydub>=0.25.1",
  "faster-whisper>=1.0.3",
  "paddleocr>=2.7.3",
```

(Removed: `openai`, `edge-tts`. Added: `anthropic`, `httpx`.)

- [ ] **Step 3: Install new dependencies**

Run: `pip install -e ".[dev]"`
Expected: anthropic and httpx install successfully; openai and edge-tts are no longer required by the project (they may still be in env but are unused).

- [ ] **Step 4: Run full test suite — expect failures in translate/tts/pipeline tests**

Run: `python -m pytest 2>&1 | tail -5`
Expected: Many failures, all in `test_translate.py`, `test_translate_responses.py`, `test_pipeline.py`. Settings import succeeds; failures are downstream.

- [ ] **Step 5: Commit**

```bash
git add src/vietdub/config.py pyproject.toml
git commit -m "chore: rename OpenAI settings to Anthropic, add MiniMax TTS fields"
```

---

### Task 2: Rewrite translate.py with anthropic SDK (TDD)

**Files:**
- Modify: `src/vietdub/translate.py`
- Modify: `tests/test_translate_responses.py` (full rewrite)
- Modify: `tests/test_translate.py` (add error-path tests)
- Existing tests `test_review_csv_*`, `test_translation_prompt_*`, `test_parse_translation_response_*`, `test_import_review_csv_*` keep working unchanged.

**Interfaces:**
- Consumes: `Settings` from Task 1; new `anthropic` SDK
- Produces: `translate_with_llm(segments, context_bundle, *, settings) -> list[TranslationRow]` — **new signature**: takes a `Settings` object instead of separate `api_key/model/base_url` kwargs. The `pipeline.py` caller will be updated in Task 4.

Public function signature change: `translate_with_llm(segments, context_bundle, *, settings)` replacing `(segments, context_bundle, api_key, model, base_url="")`. This is intentional and called out in spec.

- [ ] **Step 1: Write new test in `tests/test_translate_responses.py` (full rewrite)**

Replace the entire file with:

```python
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
```

- [ ] **Step 2: Run new tests — expect failures (function not migrated yet)**

Run: `python -m pytest tests/test_translate_responses.py -v 2>&1 | tail -10`
Expected: All 6 tests FAIL — `translate_with_llm` still has old signature `(segments, context_bundle, api_key, model, base_url="")` and uses `openai`. Tests call new signature with `settings=`.

- [ ] **Step 3: Add H4 fix to `parse_translation_response` (H4 = log warning on invalid status)**

In `src/vietdub/translate.py`, find the block:

```python
if item.get("status") not in ALLOWED_REVIEW_STATUSES:
    item = {**item, "status": "draft"}
```

Replace with:

```python
if item.get("status") not in ALLOWED_REVIEW_STATUSES:
    print(
        f"Warning: LLM returned invalid status {item.get('status')!r} "
        f"for segment {item.get('segment_id')!r}; coercing to 'draft'",
        file=sys.stderr,
    )
    item = {**item, "status": "draft"}
```

And add `import sys` to the top of the file (next to existing imports).

- [ ] **Step 4: Rewrite `translate_with_llm` in `src/vietdub/translate.py`**

Find the entire `translate_with_llm` function (lines 69-115) and replace it with:

```python
def translate_with_llm(
    segments: list[TimedSegment],
    context_bundle: dict,
    *,
    settings,
) -> list[TranslationRow]:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for translation")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is required for translation")

    import anthropic

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
    system_prompt = (
        "You are a Vietnamese localization editor for Chinese comedy cartoons. "
        "Return JSON only with a top-level \"translations\" array."
    )
    user_prompt = build_translation_prompt(segments, context_bundle)

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(f"MiniMax authentication failed: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"MiniMax HTTP {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(f"MiniMax network error: {exc}") from exc

    content_blocks = response.content or []
    text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise RuntimeError("MiniMax returned empty response (no text content blocks)")
    content = "".join(text_parts)
    return parse_translation_response(content)
```

- [ ] **Step 5: Run new tests — expect pass**

Run: `python -m pytest tests/test_translate_responses.py -v 2>&1 | tail -10`
Expected: 6/6 passing.

- [ ] **Step 6: Run full suite — expect failures only in pipeline tests (TTS, edge_voice references)**

Run: `python -m pytest 2>&1 | tail -5`
Expected: Translate tests pass; failures in `test_pipeline.py` and any other test that calls `translate_with_llm` with old signature.

- [ ] **Step 7: Commit**

```bash
git add src/vietdub/translate.py tests/test_translate.py tests/test_translate_responses.py
git commit -m "fix: switch LLM client from OpenAI SDK to anthropic SDK targeting MiniMax"
```

---

### Task 3: Replace EdgeTtsEngine with MiniMaxTtsEngine (TDD)

**Files:**
- Modify: `src/vietdub/tts.py`
- Create: `tests/test_tts.py`
- (Old `tests/test_tts.py` does not exist.)

**Interfaces:**
- Consumes: `Settings` from Task 1; `httpx`
- Produces: `class MiniMaxTtsEngine` with signature `__init__(self, voice_id: str, api_key: str, base_url: str, model: str = "speech-2.8-hd", timeout: float = 30.0)` and `async def synthesize_segment(self, row, output) -> Path`

`EdgeTtsEngine` is removed entirely.

- [ ] **Step 1: Create `tests/test_tts.py` with full test suite**

```python
import json
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
            text=json.dumps(self.response_payload),
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
```

- [ ] **Step 2: Run new tests — expect import failure**

Run: `python -m pytest tests/test_tts.py -v 2>&1 | tail -10`
Expected: FAIL — `MiniMaxTtsEngine` does not exist yet.

- [ ] **Step 3: Replace `src/vietdub/tts.py`**

Replace the entire file with:

```python
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
```

- [ ] **Step 4: Run new tests — expect pass**

Run: `python -m pytest tests/test_tts.py -v 2>&1 | tail -10`
Expected: 3/3 passing.

- [ ] **Step 5: Run full suite — expect failures in test_pipeline.py**

Run: `python -m pytest 2>&1 | tail -5`
Expected: Translate + TTS tests pass. Failures in `test_pipeline.py` from `EdgeTtsEngine` not existing and `edge_voice` field removed.

- [ ] **Step 6: Commit**

```bash
git add src/vietdub/tts.py tests/test_tts.py
git commit -m "feat: replace EdgeTtsEngine with MiniMaxTtsEngine (httpx)"
```

---

### Task 4: Wire MiniMaxTtsEngine in pipeline + update test_pipeline.py

**Files:**
- Modify: `src/vietdub/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `MiniMaxTtsEngine` from Task 3; new `Settings` from Task 1
- Produces: `pipeline.py:resume_tts_and_render` instantiates `MiniMaxTtsEngine` from `settings.tts_*` fields. All `test_pipeline.py` monkeypatches and settings stubs updated to match.

- [ ] **Step 1: Update `src/vietdub/pipeline.py` — TTS engine import and instantiation**

In `src/vietdub/pipeline.py`, find:

```python
from .tts import EdgeTtsEngine
```

Replace with:

```python
from .tts import MiniMaxTtsEngine
```

Then find (around line 243):

```python
        engine = EdgeTtsEngine(settings.edge_voice)
```

Replace with:

```python
        engine = MiniMaxTtsEngine(
            voice_id=settings.tts_voice_id,
            api_key=settings.tts_api_key,
            base_url=settings.tts_base_url,
            model=settings.tts_model,
        )
```

- [ ] **Step 2: Update `translate_with_llm` call in `src/vietdub/pipeline.py`**

In `src/vietdub/pipeline.py`, find the call to `translate_with_llm` (around line 100). The current call passes `api_key`, `model`, `base_url` kwargs. Replace those with a `settings=settings` kwarg.

Before (typical pattern):

```python
    translations = translate_with_llm(
        segments=vietnamese_segments,
        context_bundle=context_bundle,
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
    )
```

After:

```python
    translations = translate_with_llm(
        segments=vietnamese_segments,
        context_bundle=context_bundle,
        settings=settings,
    )
```

(Adjust the exact kwargs being passed based on the current code; the rule is: drop `api_key`, `model`, `base_url` and add `settings=settings`.)

- [ ] **Step 3: Update `_resume_settings()` helper in `tests/test_pipeline.py`**

Find the helper near the top of `tests/test_pipeline.py`:

```python
def _resume_settings():
    return type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural", "sample_rate": 44_100})()
```

Replace with:

```python
def _resume_settings():
    return type(
        "Settings",
        (),
        {
            "anthropic_api_key": "test-key",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "llm_model": "MiniMax-M3",
            "tts_voice_id": "vi-VN-HoaiMyNeural",
            "tts_api_key": "test-tts-key",
            "tts_base_url": "https://api.minimax.io/v1",
            "tts_model": "speech-2.8-hd",
            "sample_rate": 44_100,
        },
    )()
```

- [ ] **Step 4: Bulk-replace all `EdgeTtsEngine` monkeypatches**

In `tests/test_pipeline.py`, replace all occurrences of:

```python
monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", ...)
```

with:

```python
monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", ...)
```

Use Edit with `replace_all=True` (single string, no surrounding context — should appear 10+ times in this file).

- [ ] **Step 5: Bulk-replace all `{"edge_voice": ...}` inline stubs**

In `tests/test_pipeline.py`, replace all inline settings stubs that use `edge_voice` with the new field set. Search for `{"edge_voice":` (literal substring) and replace each occurrence with:

```python
{
    "anthropic_api_key": "test-key",
    "anthropic_base_url": "https://api.minimax.io/anthropic",
    "llm_model": "MiniMax-M3",
    "tts_voice_id": "vi-VN-HoaiMyNeural",
    "tts_api_key": "test-tts-key",
    "tts_base_url": "https://api.minimax.io/v1",
    "tts_model": "speech-2.8-hd",
    "sample_rate": 44_100,
}
```

There are ~10 such inline stubs in the file. Use Edit with `replace_all=True` (the inline stub text is identical at each site).

- [ ] **Step 6: Update `translate_with_llm` mock site in `test_pipeline.py`**

In `tests/test_pipeline.py`, the existing test that monkeypatches `vietdub.translate.translate_with_llm` (around line 724) does NOT need code changes — the mock just replaces the function entirely. Verify it still works.

- [ ] **Step 7: Run full suite — expect all green**

Run: `python -m pytest 2>&1 | tail -5`
Expected: 100+ passed, 0 failed. The 1 warning (pydub audioop deprecation) remains.

- [ ] **Step 8: Commit**

```bash
git add src/vietdub/pipeline.py tests/test_pipeline.py
git commit -m "fix: wire MiniMaxTtsEngine and updated settings into pipeline"
```

---

### Task 5: Update README setup section

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: completed migration from Tasks 1-4
- Produces: README with new env vars, MiniMax API key instructions, voice_id lookup note

- [ ] **Step 1: Find and update the Setup / Configuration section in `README.md`**

Locate the section that currently mentions `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`, and any edge_tts / `vi-VN-HoaiMyNeural` instructions. Replace with the following text (adapt section heading to match existing README style):

````markdown
## Configuration

Create a `.env` file in the project root with:

```ini
# MiniMax (LLM + TTS) — get an API key from https://platform.minimax.io
ANTHROPIC_API_KEY=sk-...
LLM_MODEL=MiniMax-M3
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic

# MiniMax TTS — pick a voice_id from https://platform.minimax.io/faq/system-voice-id
TTS_API_KEY=sk-...                  # can be the same as ANTHROPIC_API_KEY
TTS_VOICE_ID=vi-female-1
TTS_MODEL=speech-2.8-hd
TTS_BASE_URL=https://api.minimax.io/v1/t2a_v2
```

**Note:** `TTS_VOICE_ID` must be set to a valid MiniMax voice_id before first use. The default in `Settings` (`vi-VN-HoaiMyNeural`) is the legacy edge_tts voice name and will fail at first TTS call. Look up available voices at `https://platform.minimax.io/faq/system-voice-id`.

TTS calls are billed per character (was free with edge_tts in earlier versions). LLM calls are billed per token.
````

(If the existing README uses different section names or structure, adapt the heading and placement but keep the content: MiniMax API key, voice_id lookup, billing note.)

- [ ] **Step 2: Update any other mentions of OpenAI, edge_tts, or voice names throughout README**

Search `README.md` for: `OpenAI`, `openai`, `edge_tts`, `edge-tts`, `vi-VN-HoaiMyNeural` (when used as a config example, not as a placeholder default), `OPENAI_API_KEY`. For each, either:
- Remove the reference (if obsolete)
- Replace with the MiniMax equivalent (if it's a config example)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README setup for MiniMax LLM + TTS"
```

---

## Self-Review Notes

- **Spec coverage:**
  - Section 1 (Config) → Task 1
  - Section 2 (translate.py rewrite) → Task 2
  - Section 3 (tts.py rewrite) → Task 4 → wait, Task 3 (TTS engine) + Task 4 (pipeline wiring)
  - Section 4 (Test changes) → Tasks 2, 3, 4
  - Section 5 (README) → Task 5
  - Section 6 (Dependencies) → Task 1
  - H4 fix → Task 2 Step 3
- **No placeholders:** every code block is complete; settings dict literal is the same at every site
- **Type consistency:** `MiniMaxTtsEngine.__init__` signature and `translate_with_llm` signature defined in Task 2/3 are referenced by Task 4 exactly
- **Shared dict literal:** the inline settings stub for `Settings` is repeated verbatim at ~10 sites in `test_pipeline.py`. This is intentional and addresses the "No 'similar to Task N' (repeat the code)" rule.
