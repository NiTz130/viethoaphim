# MiniMax LLM + TTS Migration Design

**Date:** 2026-06-27
**Scope:** Replace LLM (OpenAI) and TTS (edge_tts) engines with MiniMax APIs
**Status:** Approved (design), pending implementation

## Context

Vietdub currently uses two external services for translation and speech synthesis:

- **LLM**: OpenAI Python SDK (chat completions format), configured via `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- **TTS**: `edge_tts` (Microsoft, free, no auth)

This spec replaces both with MiniMax APIs:

- **LLM**: `MiniMax-M3` (multimodal, 1M context) via Anthropic-compatible endpoint
- **TTS**: `speech-2.8-hd` REST API

OCR (PaddleOCR) and STT (Faster-Whisper) are unchanged — MiniMax has no equivalent product.

## Goals

- Replace OpenAI SDK with `anthropic` SDK targeting MiniMax's Anthropic-compatible endpoint
- Replace `edge_tts` with MiniMax Speech 2.8 REST API
- Add error-path test coverage for the new translation client (addresses M8 from prior review)
- Preserve current public function signatures so consumers in `pipeline.py` are unchanged

## Non-goals

- OCR engine replacement (PaddleOCR stays)
- STT engine replacement (Faster-Whisper stays)
- OpenAI backward compatibility — hard switch
- Multi-provider abstraction layer
- Cost tracking or budget enforcement
- Async batching of TTS requests (current code already calls per-segment)

## Design

### 1. Config changes (`src/vietdub/config.py`)

Rename env fields and add TTS fields:

```python
class Settings(BaseSettings):
    # LLM (MiniMax via Anthropic-compatible endpoint)
    anthropic_api_key: str = ""                              # was openai_api_key
    anthropic_base_url: str = "https://api.minimax.io/anthropic"  # was openai_base_url
    llm_model: str = "MiniMax-M3"                            # was "" (empty)

    # TTS (MiniMax Speech 2.8)
    tts_voice_id: str = "vi-VN-HoaiMyNeural"                 # NEW; edge_tts name as placeholder
    tts_model: str = "speech-2.8-hd"                         # NEW
    tts_base_url: str = "https://api.minimax.io/v1/t2a_v2"    # NEW
    tts_api_key: str = ""                                     # NEW; defaults empty
    tts_timeout: float = 30.0                                 # NEW; per-segment HTTP timeout

    # Unchanged
    sample_rate: int = 44_100
    reference_data_dir: str = "data"
```

**Removed**: `edge_voice` (no longer used).

**Hard rename**: no `OPENAI_API_KEY` alias. Users must update `.env` explicitly.

### 2. `translate.py` rewrite

Switch from `openai` SDK to `anthropic` SDK. Public function signature unchanged.

**Imports:**
```python
import anthropic
from .models import TranslationRow
```

**New client construction:**
```python
client = anthropic.Anthropic(
    api_key=settings.anthropic_api_key,
    base_url=settings.anthropic_base_url,
)
```

**New call shape:**
```python
response = client.messages.create(
    model=settings.llm_model,
    max_tokens=4096,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}],
)
```

**Response parsing**: `response.content` is a list of content blocks. Extract text by joining `block.text` for each `TextBlock` in the list. Skip `ThinkingBlock`s.

**Error mapping:**
| Anthropic exception | RuntimeError message |
|---|---|
| `anthropic.AuthenticationError` | `"MiniMax authentication failed: ..."` |
| `anthropic.APIStatusError` | `"MiniMax HTTP {status_code}: ..."` |
| `anthropic.APIConnectionError` | `"MiniMax network error: ..."` |
| (any other APIError) | `"MiniMax API error: ..."` |
| Empty content list | `"MiniMax returned empty response"` |

**Status downgrade behavior (H4 from prior review)**: when LLM returns a row with `status` not in `{"draft", "reviewed", "skip"}`, log a warning to stderr and accept as `"draft"`. This preserves the prior permissive default but surfaces the model misbehavior. Alternative "reject and raise" was rejected as too disruptive — current code's pydantic validator would also reject invalid statuses, so a stricter policy here would be a behavior change.

### 3. `tts.py` rewrite (edge_tts → MiniMax Speech 2.8)

Replace `EdgeTtsEngine` with `MiniMaxTtsEngine` using `httpx.AsyncClient`.

**New class signature:**
```python
class MiniMaxTtsEngine:
    def __init__(self, voice_id: str, api_key: str, base_url: str, model: str = "speech-2.8-hd"):
        ...

    async def synthesize_segment(self, row: TranslationRow, output: Path) -> Path:
        ...
```

**Request payload:**
```python
{
    "model": self.model,
    "text": row.text_vi,
    "voice_setting": {"voice_id": self.voice_id, "language_boost": "auto"},
    "audio_setting": {"format": "mp3", "sample_rate": 24000},
    "output_format": "hex",
}
```

**Response handling:**
- HTTP 200 + `base_resp.status_code == 0` → decode `data.audio` (hex) to bytes, write to `output`
- HTTP error → raise `RuntimeError` with status code and body
- Network error → raise `RuntimeError`
- `base_resp.status_code != 0` → raise `RuntimeError` with `status_msg`

**Caller wiring** (`pipeline.py:resume_tts_and_render`):
```python
engine = MiniMaxTtsEngine(
    voice_id=settings.tts_voice_id,
    api_key=settings.tts_api_key,
    base_url=settings.tts_base_url,
    model=settings.tts_model,
)
```

**Removed**: `EdgeTtsEngine` class entirely.

### 4. Test changes

**`tests/test_translate.py` + `test_translate_responses.py`:**
- Mock `anthropic.Anthropic` (via `monkeypatch.setattr("vietdub.translate.anthropic.Anthropic", ...)`)
- Mock response is a `Message`-like object with `content=[TextBlock(text="...")]`
- New error-path tests (addresses M8 from prior review):
  - `anthropic.AuthenticationError` → `RuntimeError("MiniMax authentication failed")`
  - `anthropic.APIStatusError` (e.g. 429, 500) → `RuntimeError("MiniMax HTTP {code}")`
  - `anthropic.APIConnectionError` → `RuntimeError("MiniMax network error")`
  - Empty content list → `RuntimeError("MiniMax returned empty response")`
  - Valid content but unparseable JSON → existing behavior preserved

**`tests/test_pipeline.py`:**
- Existing `monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", ...)` patches → update target to `vietdub.tts.MiniMaxTtsEngine.synthesize_segment`
- New helper `_make_async_minimax_tts_mock()` returns a `synthesize_segment` that writes a valid MP3 (reuse `_write_valid_mp3`)

**New file `tests/test_tts.py`:**
- Test `MiniMaxTtsEngine.synthesize_segment` happy path (mock httpx, verify payload + hex decode)
- Test HTTP error → RuntimeError
- Test `base_resp.status_code != 0` → RuntimeError
- Test network error → RuntimeError
- Test missing `tts_api_key` → informative error

### 5. Documentation updates

**`README.md` Setup section:**
- Remove OpenAI setup instructions
- Add: get API key from `platform.minimax.io`, set `ANTHROPIC_API_KEY` in `.env`
- Add: look up Vietnamese voice_id from `https://platform.minimax.io/faq/system-voice-id`, set `TTS_VOICE_ID`
- Note: TTS is now paid API (was free with edge_tts)
- Update first-run notes (model download no longer applies to LLM; STT model still downloads)

### 6. Dependencies

**`pyproject.toml`:**
- Add: `anthropic>=0.40.0` (current as of 2026)
- Add: `httpx>=0.27.0` (likely already present; verify)
- Remove: `openai` (no longer used)
- Keep: `edge-tts` (if listed; remove since unused)

## File inventory

**Modified (9 source + 1 new test = 10 files):**
1. `src/vietdub/config.py` — Settings field rename + new TTS fields
2. `src/vietdub/translate.py` — anthropic SDK rewrite
3. `src/vietdub/tts.py` — MiniMax HTTP rewrite
4. `src/vietdub/pipeline.py` — caller wiring (instantiate `MiniMaxTtsEngine` instead of `EdgeTtsEngine`)
5. `tests/test_translate.py` — mock updates + new error tests
6. `tests/test_translate_responses.py` — mock updates
7. `tests/test_pipeline.py` — monkeypatch target update
8. `tests/test_tts.py` — NEW file for TTS engine unit tests
9. `README.md` — setup section
10. `pyproject.toml` — dependency changes

**Unchanged (no LLM/TTS touchpoints):**
`media.py`, `sync.py`, `ocr.py`, `stt.py`, `jobs.py`, `merge.py`, `context.py`, `memory.py`, `reference.py`, `srt.py`, `models.py`, `cli.py`

## Risk

- **API cost**: edge_tts was free; MiniMax Speech 2.8 is paid. First-time users will incur TTS API charges on real jobs.
- **Vietnamese TTS quality**: unknown until tested against real jobs. Mitigation: speech-2.8-hd is the high-quality variant; users can downgrade to turbo.
- **Voice ID default**: `"vi-VN-HoaiMyNeural"` is the edge_tts name and almost certainly invalid for MiniMax. First real call will fail with a clear API error. Mitigation: README clearly instructs to look up voice_id; fail-fast validation could be added but deferred.
- **API rate limits / latency**: HTTP calls add latency vs edge_tts. With per-segment TTS, large jobs (hundreds of segments) could hit rate limits. Mitigation: 30s per-request timeout ensures the pipeline doesn't hang; rate-limit handling is deferred to a future iteration.
- **Multimodal thinking blocks**: MiniMax M3 may return `ThinkingBlock`s in addition to `TextBlock`s. Spec requires extracting only text, but if M3's thinking is critical for translation quality, behavior may differ from OpenAI models. Mitigation: implicit in Anthropic SDK design; verify with real translation tests.

## Rollback

Single migration — revert the commits. Since this is a hard rename (no backward compat), rollback is straightforward via `git revert`.

## Implementation notes

- Order of commits: config first (breaks `Settings()` consumers until they update), then `translate.py`, then `tts.py`, then `pipeline.py` wiring, then tests, then docs. The pipeline.py change is the one that actually unblocks the end-to-end flow.
- A useful split: one commit per file modified. Keeps review surface small per commit.
- `tts_voice_id` default is intentionally the edge_tts name so the first run produces a clear, recognizable error rather than a silent fallback to a random MiniMax default.
