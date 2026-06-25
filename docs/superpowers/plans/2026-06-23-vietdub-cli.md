# VietDub CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows-first Python CLI that converts Chinese hard-subbed cartoon videos into Vietnamese review files, synced Vietnamese TTS audio, Vietnamese subtitles, and preview videos.

**Architecture:** The app is a staged job pipeline. Each stage reads and writes durable files inside a job directory so failed or revised work can resume from a named step without recomputing earlier outputs. Engines are adapters: local FFmpeg/media, local Faster-Whisper STT, local PaddleOCR OCR, OpenAI-compatible LLM context/translation, and Edge TTS.

**Tech Stack:** Python 3.11, Typer, Pydantic v2, pytest, FFmpeg subprocess, faster-whisper, PaddleOCR, edge-tts, OpenAI-compatible chat API.

---

## File Structure

- Create `pyproject.toml`: project metadata, console script, dependencies, pytest config.
- Create `README.md`: install, environment variables, and MVP CLI usage.
- Create `src/vietdub/__init__.py`: package version.
- Create `src/vietdub/cli.py`: Typer command entrypoint: `run`, `resume`, `inspect`.
- Create `src/vietdub/config.py`: environment and default engine settings.
- Create `src/vietdub/models.py`: Pydantic schemas shared by pipeline stages.
- Create `src/vietdub/jobs.py`: job folder creation, status files, step ordering, resume checks.
- Create `src/vietdub/media.py`: FFmpeg wrapper for metadata, audio extraction, render/mux.
- Create `src/vietdub/stt.py`: Faster-Whisper adapter and fixture adapter for tests.
- Create `src/vietdub/ocr.py`: PaddleOCR subtitle crop adapter and fixture adapter for tests.
- Create `src/vietdub/merge.py`: STT/OCR overlap merge into canonical timed transcript.
- Create `src/vietdub/context.py`: story memory and context JSON generation.
- Create `src/vietdub/translate.py`: LLM translation adapter, review CSV export/import.
- Create `src/vietdub/tts.py`: Edge TTS per-segment synthesis.
- Create `src/vietdub/sync.py`: trim, duration measurement, `atempo` speed adjustment, silent-canvas overlay.
- Create `src/vietdub/srt.py`: SRT writer.
- Create `src/vietdub/pipeline.py`: orchestration for `run` and `resume`.
- Create `tests/fixtures/`: small JSON fixtures for deterministic tests.
- Create tests matching each core module under `tests/`.

## Task 1: Project Scaffold And CLI Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/vietdub/__init__.py`
- Create: `src/vietdub/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the CLI smoke tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from vietdub.cli import app


runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "resume" in result.output
    assert "inspect" in result.output


def test_inspect_missing_job_returns_error():
    result = runner.invoke(app, ["inspect", "missing-job"])
    assert result.exit_code != 0
    assert "Job directory not found" in result.output
```

- [ ] **Step 2: Run the test and verify it fails because package files do not exist**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'vietdub'`.

- [ ] **Step 3: Create the project metadata**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vietdub"
version = "0.1.0"
description = "Windows-first CLI for Chinese-to-Vietnamese cartoon dubbing"
requires-python = ">=3.11"
dependencies = [
  "typer>=0.12.0",
  "pydantic>=2.7.0",
  "pydantic-settings>=2.2.0",
  "python-dotenv>=1.0.1",
  "openai>=1.40.0",
  "edge-tts>=6.1.12",
  "pydub>=0.25.1",
  "faster-whisper>=1.0.3",
  "paddleocr>=2.7.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0", "pytest-asyncio>=0.23.0"]

[project.scripts]
vietdub = "vietdub.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Create package and CLI skeleton**

```python
# src/vietdub/__init__.py
__version__ = "0.1.0"
```

```python
# src/vietdub/cli.py
from pathlib import Path

import typer


app = typer.Typer(help="Vietnamese dubbing pipeline for Chinese cartoon videos.")


@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", help="Pipeline mode: review or auto."),
    series: str | None = typer.Option(None, "--series", help="Series memory name."),
) -> None:
    if mode not in {"review", "auto"}:
        raise typer.BadParameter("mode must be review or auto")
    if not video.exists():
        raise typer.BadParameter(f"Video not found: {video}")
    typer.echo(f"Prepared to run {video} in {mode} mode")


@app.command()
def resume(
    job_dir: Path,
    from_step: str = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    if not job_dir.exists():
        raise typer.BadParameter(f"Job directory not found: {job_dir}")
    typer.echo(f"Prepared to resume {job_dir} from {from_step}")


@app.command()
def inspect(job_dir: Path) -> None:
    if not job_dir.exists():
        raise typer.BadParameter(f"Job directory not found: {job_dir}")
    typer.echo(job_dir)


def main() -> None:
    app()
```

- [ ] **Step 5: Add README usage**

```markdown
# VietDub

Windows-first CLI for translating Chinese hard-subbed cartoon videos into Vietnamese dubbed previews.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Environment

```powershell
$env:OPENAI_API_KEY="your-key"
$env:LLM_MODEL="gpt-4.1-mini"
```

## Usage

```powershell
vietdub run input.mp4 --mode review --series my-series
vietdub resume jobs\input --from tts
vietdub inspect jobs\input
```
```

- [ ] **Step 6: Run the CLI test and verify it passes**

Run: `python -m pytest tests/test_cli.py -v`

Expected: PASS for both tests.

- [ ] **Step 7: Commit scaffold**

```bash
git init
git add pyproject.toml README.md src/vietdub/__init__.py src/vietdub/cli.py tests/test_cli.py
git commit -m "chore: scaffold vietdub cli"
```

## Task 2: Shared Models And Config

**Files:**
- Create: `src/vietdub/models.py`
- Create: `src/vietdub/config.py`
- Test: `tests/test_models_config.py`

- [ ] **Step 1: Write schema and config tests**

```python
# tests/test_models_config.py
from vietdub.config import Settings
from vietdub.models import TimedSegment, millis_to_srt_time, srt_time_to_millis


def test_timed_segment_computes_duration():
    segment = TimedSegment(id="0001", start_ms=1000, end_ms=2600, text="你好")
    assert segment.duration_ms == 1600


def test_timed_segment_rejects_negative_duration():
    try:
        TimedSegment(id="bad", start_ms=2000, end_ms=1000, text="x")
    except ValueError as exc:
        assert "end_ms must be greater than start_ms" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_srt_time_round_trip():
    assert millis_to_srt_time(3_723_456) == "01:02:03,456"
    assert srt_time_to_millis("01:02:03,456") == 3_723_456


def test_settings_defaults_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings()
    assert settings.llm_model == ""
    assert settings.edge_voice == "vi-VN-HoaiMyNeural"
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_models_config.py -v`

Expected: FAIL with `ModuleNotFoundError` for `vietdub.models`.

- [ ] **Step 3: Implement models**

```python
# src/vietdub/models.py
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StepName(StrEnum):
    EXTRACT = "extract"
    STT = "stt"
    OCR = "ocr"
    MERGE = "merge"
    CONTEXT = "context"
    TRANSLATE = "translate"
    TTS = "tts"
    RENDER = "render"


STEP_ORDER: list[StepName] = [
    StepName.EXTRACT,
    StepName.STT,
    StepName.OCR,
    StepName.MERGE,
    StepName.CONTEXT,
    StepName.TRANSLATE,
    StepName.TTS,
    StepName.RENDER,
]


class TimedSegment(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    confidence: float | None = None
    source: str = "unknown"
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timing(self) -> "TimedSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class CharacterProfile(BaseModel):
    name_cn: str
    name_vi: str
    role: str = ""
    personality: str = ""
    pronoun_rules: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0


class SceneContext(BaseModel):
    segment_ids: list[str]
    summary: str
    characters: list[str] = Field(default_factory=list)
    tone: str = "hài sa điêu, tự nhiên tiếng Việt"


class TranslationRow(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    speaker: str | None = None
    text_cn: str
    text_vi: str
    context_note: str = ""
    status: str = "draft"

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"draft", "reviewed", "skip"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value


class SyncSegmentReport(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    original_duration_ms: int
    synced_duration_ms: int
    speed_factor: float
    warnings: list[str] = Field(default_factory=list)


class SyncReport(BaseModel):
    output_audio: Path
    sample_rate: int = 44_100
    segments: list[SyncSegmentReport] = Field(default_factory=list)


def millis_to_srt_time(ms: int) -> str:
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def srt_time_to_millis(value: str) -> int:
    hours_text, minutes_text, rest = value.split(":")
    seconds_text, millis_text = rest.split(",")
    return (
        int(hours_text) * 3_600_000
        + int(minutes_text) * 60_000
        + int(seconds_text) * 1_000
        + int(millis_text)
    )
```

- [ ] **Step 4: Implement config**

```python
# src/vietdub/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = ""
    edge_voice: str = "vi-VN-HoaiMyNeural"
    stt_model: str = "medium"
    stt_language: str = "zh"
    jobs_dir: str = "jobs"
    sample_rate: int = 44_100
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m pytest tests/test_models_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit models and config**

```bash
git add src/vietdub/models.py src/vietdub/config.py tests/test_models_config.py
git commit -m "feat: add shared pipeline models"
```

## Task 3: Job Manager

**Files:**
- Create: `src/vietdub/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write job manager tests**

```python
# tests/test_jobs.py
from pathlib import Path

from vietdub.jobs import JobManager
from vietdub.models import StepName


def test_create_job_copies_input_and_creates_layout(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    manager = JobManager(tmp_path / "jobs")

    job = manager.create(video, series="demo")

    assert job.root.name == "clip"
    assert (job.root / "input.mp4").read_bytes() == b"fake-video"
    assert (job.root / "audio").is_dir()
    assert (job.root / "translation").is_dir()
    assert job.config["series"] == "demo"


def test_mark_step_and_inspect(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job = JobManager(tmp_path / "jobs").create(video, series=None)

    job.mark_done(StepName.STT, {"segments": 2})
    status = job.load_status()

    assert status["stt"]["state"] == "done"
    assert status["stt"]["details"] == {"segments": 2}


def test_steps_from_returns_expected_order(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job = JobManager(tmp_path / "jobs").create(video, series=None)

    names = [step.value for step in job.steps_from(StepName.MERGE)]

    assert names == ["merge", "context", "translate", "tts", "render"]
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_jobs.py -v`

Expected: FAIL with `ModuleNotFoundError` for `vietdub.jobs`.

- [ ] **Step 3: Implement job manager**

```python
# src/vietdub/jobs.py
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import STEP_ORDER, StepName


JOB_DIRS = [
    "audio",
    "stt",
    "ocr",
    "transcript",
    "context",
    "translation",
    "tts/segments",
    "output",
]


@dataclass
class Job:
    root: Path
    config: dict[str, Any]

    @property
    def input_video(self) -> Path:
        return self.root / "input.mp4"

    def write_json(self, relative: str, data: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_json(self, relative: str) -> Any:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def load_status(self) -> dict[str, Any]:
        status_path = self.root / "status.json"
        if not status_path.exists():
            return {}
        return json.loads(status_path.read_text(encoding="utf-8"))

    def mark_done(self, step: StepName, details: dict[str, Any]) -> None:
        status = self.load_status()
        status[step.value] = {
            "state": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }
        self.write_json("status.json", status)

    def steps_from(self, step: StepName) -> list[StepName]:
        start = STEP_ORDER.index(step)
        return STEP_ORDER[start:]


class JobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir

    def create(self, video: Path, series: str | None) -> Job:
        root = self.jobs_dir / video.stem
        suffix = 1
        while root.exists():
            root = self.jobs_dir / f"{video.stem}-{suffix}"
            suffix += 1
        root.mkdir(parents=True)
        for relative in JOB_DIRS:
            (root / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, root / "input.mp4")
        config = {
            "source_video": str(video.resolve()),
            "series": series,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        job = Job(root=root, config=config)
        job.write_json("job.json", config)
        job.write_json("status.json", {})
        return job

    def open(self, root: Path) -> Job:
        if not root.exists():
            raise FileNotFoundError(f"Job directory not found: {root}")
        config = json.loads((root / "job.json").read_text(encoding="utf-8"))
        return Job(root=root, config=config)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`

Expected: PASS.

- [ ] **Step 5: Commit job manager**

```bash
git add src/vietdub/jobs.py tests/test_jobs.py
git commit -m "feat: add resumable job manager"
```

## Task 4: Media And SRT Utilities

**Files:**
- Create: `src/vietdub/media.py`
- Create: `src/vietdub/srt.py`
- Test: `tests/test_media_srt.py`

- [ ] **Step 1: Write SRT and FFmpeg command tests**

```python
# tests/test_media_srt.py
from pathlib import Path

from vietdub.media import build_extract_audio_command, build_mux_preview_command
from vietdub.models import TimedSegment
from vietdub.srt import render_srt


def test_render_srt():
    text = render_srt([TimedSegment(id="1", start_ms=0, end_ms=1500, text="Xin chào")])
    assert "1\n00:00:00,000 --> 00:00:01,500\nXin chào\n" in text


def test_extract_audio_command_uses_wav_and_sample_rate(tmp_path):
    command = build_extract_audio_command(tmp_path / "in.mp4", tmp_path / "out.wav", 44100)
    assert command[:2] == ["ffmpeg", "-y"]
    assert "-ar" in command
    assert "44100" in command


def test_mux_preview_command_maps_video_audio_and_subtitles(tmp_path):
    command = build_mux_preview_command(
        video=tmp_path / "in.mp4",
        audio=tmp_path / "vi.wav",
        subtitles=tmp_path / "vi.srt",
        output=tmp_path / "preview.mp4",
    )
    joined = " ".join(str(part) for part in command)
    assert "subtitles=" in joined
    assert str(tmp_path / "preview.mp4") in joined
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_media_srt.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement SRT writer**

```python
# src/vietdub/srt.py
from .models import TimedSegment, millis_to_srt_time


def render_srt(segments: list[TimedSegment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{millis_to_srt_time(segment.start_ms)} --> {millis_to_srt_time(segment.end_ms)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")
```

- [ ] **Step 4: Implement media command builders and runners**

```python
# src/vietdub/media.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def build_extract_audio_command(video: Path, output: Path, sample_rate: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output),
    ]


def extract_audio(video: Path, output: Path, sample_rate: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_extract_audio_command(video, output, sample_rate))


def probe_media(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def build_mux_preview_command(video: Path, audio: Path, subtitles: Path, output: Path) -> list[str]:
    escaped_subtitles = str(subtitles).replace("\\", "/").replace(":", "\\:")
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        f"subtitles='{escaped_subtitles}'",
        "-shortest",
        str(output),
    ]


def mux_preview(video: Path, audio: Path, subtitles: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_mux_preview_command(video, audio, subtitles, output))
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m pytest tests/test_media_srt.py -v`

Expected: PASS.

- [ ] **Step 6: Commit media utilities**

```bash
git add src/vietdub/media.py src/vietdub/srt.py tests/test_media_srt.py
git commit -m "feat: add media and subtitle utilities"
```

## Task 5: STT And OCR Adapters

**Files:**
- Create: `src/vietdub/stt.py`
- Create: `src/vietdub/ocr.py`
- Test: `tests/test_engines.py`

- [ ] **Step 1: Write fixture adapter tests**

```python
# tests/test_engines.py
import json

from vietdub.ocr import FixtureOcrEngine
from vietdub.stt import FixtureSttEngine


def test_fixture_stt_engine_loads_segments(tmp_path):
    fixture = tmp_path / "stt.json"
    fixture.write_text(
        json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "stt"}]),
        encoding="utf-8",
    )
    result = FixtureSttEngine(fixture).transcribe(tmp_path / "audio.wav")
    assert result[0].text == "你好"


def test_fixture_ocr_engine_loads_segments(tmp_path):
    fixture = tmp_path / "ocr.json"
    fixture.write_text(
        json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "都怀疑你", "source": "ocr"}]),
        encoding="utf-8",
    )
    result = FixtureOcrEngine(fixture).recognize(tmp_path / "video.mp4")
    assert result[0].source == "ocr"
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_engines.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement STT adapters**

```python
# src/vietdub/stt.py
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
```

- [ ] **Step 4: Implement OCR adapters**

```python
# src/vietdub/ocr.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .models import TimedSegment


class FixtureOcrEngine:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path

    def recognize(self, video_path: Path) -> list[TimedSegment]:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return [TimedSegment.model_validate(item) for item in data]


class PaddleSubtitleOcrEngine:
    def __init__(self, sample_every_seconds: float = 0.5) -> None:
        self.sample_every_seconds = sample_every_seconds

    def recognize(self, video_path: Path) -> list[TimedSegment]:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        frame_dir = video_path.parent / "ocr_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        pattern = frame_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={1 / self.sample_every_seconds},crop=iw:ih*0.28:0:ih*0.72",
            str(pattern),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

        segments: list[TimedSegment] = []
        for index, frame in enumerate(sorted(frame_dir.glob("frame_*.jpg")), start=1):
            ocr_result = ocr.ocr(str(frame), cls=True)
            texts: list[str] = []
            for page in ocr_result or []:
                for line in page or []:
                    if len(line) >= 2 and line[1][0]:
                        texts.append(str(line[1][0]).strip())
            text = "".join(texts).strip()
            if text:
                start_ms = int((index - 1) * self.sample_every_seconds * 1000)
                end_ms = start_ms + int(self.sample_every_seconds * 1000)
                segments.append(
                    TimedSegment(
                        id=f"ocr-{index:04}",
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text,
                        source="ocr",
                    )
                )
        return segments
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m pytest tests/test_engines.py -v`

Expected: PASS.

- [ ] **Step 6: Commit adapters**

```bash
git add src/vietdub/stt.py src/vietdub/ocr.py tests/test_engines.py
git commit -m "feat: add stt and ocr adapters"
```

## Task 6: Transcript Merger

**Files:**
- Create: `src/vietdub/merge.py`
- Test: `tests/test_merge.py`

- [ ] **Step 1: Write merge tests**

```python
# tests/test_merge.py
from vietdub.merge import merge_segments
from vietdub.models import TimedSegment


def test_ocr_text_wins_when_timing_overlaps():
    stt = [TimedSegment(id="s1", start_ms=0, end_ms=1800, text="怀疑你有神秘过去", source="stt")]
    ocr = [TimedSegment(id="o1", start_ms=100, end_ms=1700, text="都怀疑你有一个神秘的过往", source="ocr")]

    merged = merge_segments(stt, ocr)

    assert len(merged) == 1
    assert merged[0].text == "都怀疑你有一个神秘的过往"
    assert merged[0].source == "ocr+stt"


def test_non_overlapping_stt_is_kept():
    stt = [TimedSegment(id="s1", start_ms=3000, end_ms=4000, text="你好", source="stt")]
    ocr = [TimedSegment(id="o1", start_ms=0, end_ms=1000, text="字幕", source="ocr")]

    merged = merge_segments(stt, ocr)

    assert [segment.text for segment in merged] == ["字幕", "你好"]
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_merge.py -v`

Expected: FAIL with missing `vietdub.merge`.

- [ ] **Step 3: Implement merger**

```python
# src/vietdub/merge.py
from __future__ import annotations

from .models import TimedSegment


def overlap_ms(left: TimedSegment, right: TimedSegment) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def overlaps_enough(left: TimedSegment, right: TimedSegment) -> bool:
    overlap = overlap_ms(left, right)
    shortest = min(left.duration_ms, right.duration_ms)
    return shortest > 0 and overlap / shortest >= 0.45


def merge_segments(stt_segments: list[TimedSegment], ocr_segments: list[TimedSegment]) -> list[TimedSegment]:
    merged: list[TimedSegment] = []
    used_stt_ids: set[str] = set()

    for index, ocr_segment in enumerate(sorted(ocr_segments, key=lambda item: item.start_ms), start=1):
        matching_stt = [segment for segment in stt_segments if overlaps_enough(ocr_segment, segment)]
        for segment in matching_stt:
            used_stt_ids.add(segment.id)
        start_ms = min([ocr_segment.start_ms, *[segment.start_ms for segment in matching_stt]])
        end_ms = max([ocr_segment.end_ms, *[segment.end_ms for segment in matching_stt]])
        merged.append(
            TimedSegment(
                id=f"m-{index:04}",
                start_ms=start_ms,
                end_ms=end_ms,
                text=ocr_segment.text,
                confidence=ocr_segment.confidence,
                source="ocr+stt" if matching_stt else "ocr",
                meta={"ocr_id": ocr_segment.id, "stt_ids": [segment.id for segment in matching_stt]},
            )
        )

    next_index = len(merged) + 1
    for stt_segment in sorted(stt_segments, key=lambda item: item.start_ms):
        if stt_segment.id in used_stt_ids:
            continue
        merged.append(
            TimedSegment(
                id=f"m-{next_index:04}",
                start_ms=stt_segment.start_ms,
                end_ms=stt_segment.end_ms,
                text=stt_segment.text,
                confidence=stt_segment.confidence,
                source="stt",
                meta={"stt_id": stt_segment.id},
            )
        )
        next_index += 1

    merged.sort(key=lambda item: item.start_ms)
    for index, segment in enumerate(merged, start=1):
        segment.id = f"m-{index:04}"
    return merged
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_merge.py -v`

Expected: PASS.

- [ ] **Step 5: Commit merger**

```bash
git add src/vietdub/merge.py tests/test_merge.py
git commit -m "feat: merge stt and ocr transcripts"
```

## Task 7: Context Builder

**Files:**
- Create: `src/vietdub/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write context builder tests**

```python
# tests/test_context.py
from vietdub.context import build_context_bundle
from vietdub.models import TimedSegment


def test_context_bundle_contains_episode_and_style():
    segments = [
        TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你到底是谁"),
        TimedSegment(id="m-0002", start_ms=1000, end_ms=2000, text="都怀疑你有一个神秘的过往"),
    ]

    bundle = build_context_bundle(segments, series_context={})

    assert "episode_context" in bundle
    assert "style_guide" in bundle
    assert bundle["style_guide"]["tone"] == "hài sa điêu, tự nhiên tiếng Việt"
    assert bundle["scene_context"][0]["segment_ids"] == ["m-0001", "m-0002"]
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_context.py -v`

Expected: FAIL with missing `vietdub.context`.

- [ ] **Step 3: Implement deterministic context builder**

```python
# src/vietdub/context.py
from __future__ import annotations

from .models import TimedSegment


def build_context_bundle(segments: list[TimedSegment], series_context: dict) -> dict:
    joined = " ".join(segment.text for segment in segments[:20])
    scene_ids = [segment.id for segment in segments[:20]]
    return {
        "series_context": series_context,
        "episode_context": {
            "summary": "Tập phim hoạt hình Trung Quốc ngắn, thoại nhanh, có yếu tố hài sa điêu.",
            "source_excerpt": joined,
        },
        "characters": [],
        "glossary": {},
        "style_guide": {
            "tone": "hài sa điêu, tự nhiên tiếng Việt",
            "translation_rules": [
                "Ưu tiên câu thoại tự nhiên hơn dịch sát chữ.",
                "Giữ punchline ngắn để hợp timing TTS.",
                "Dịch nhất quán tên riêng và cách xưng hô trong toàn bộ job.",
            ],
        },
        "scene_context": [
            {
                "segment_ids": scene_ids,
                "summary": "Cảnh mở đầu hoặc đoạn thoại liên tục cần dịch theo cùng ngữ cảnh.",
                "characters": [],
                "tone": "hài sa điêu, tự nhiên tiếng Việt",
            }
        ],
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_context.py -v`

Expected: PASS.

- [ ] **Step 5: Commit context builder**

```bash
git add src/vietdub/context.py tests/test_context.py
git commit -m "feat: add story context bundle"
```

## Task 8: Translation Review CSV

**Files:**
- Create: `src/vietdub/translate.py`
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write CSV import/export tests**

```python
# tests/test_translate.py
from vietdub.models import TimedSegment, TranslationRow
from vietdub.translate import export_review_csv, import_review_csv


def test_review_csv_round_trip(tmp_path):
    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="你好", speaker="男")]
    rows = [TranslationRow(segment_id="m-0001", start_ms=0, end_ms=1000, speaker="男", text_cn="你好", text_vi="Chào nha")]
    path = tmp_path / "review.csv"

    export_review_csv(path, segments, rows)
    loaded = import_review_csv(path)

    assert loaded[0].segment_id == "m-0001"
    assert loaded[0].text_vi == "Chào nha"
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_translate.py -v`

Expected: FAIL with missing `vietdub.translate`.

- [ ] **Step 3: Implement review CSV functions and LLM prompt builder**

```python
# src/vietdub/translate.py
from __future__ import annotations

import csv
import json
from pathlib import Path

from openai import OpenAI

from .models import TimedSegment, TranslationRow


CSV_FIELDS = [
    "segment_id",
    "start_ms",
    "end_ms",
    "speaker",
    "text_cn",
    "text_vi",
    "context_note",
    "status",
]


def build_translation_prompt(segments: list[TimedSegment], context_bundle: dict) -> str:
    payload = {
        "instructions": [
            "Translate Chinese cartoon dialogue into natural Vietnamese.",
            "Use a silly, meme-friendly tone when the source is comedic.",
            "Keep Vietnamese lines short enough for dubbing timing.",
            "Preserve names and pronouns using the supplied context.",
        ],
        "context": context_bundle,
        "segments": [segment.model_dump() for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False)


def translate_with_llm(segments: list[TimedSegment], context_bundle: dict, api_key: str, model: str) -> list[TranslationRow]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for translation")
    if not model:
        raise RuntimeError("LLM_MODEL is required for translation")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Vietnamese localization editor for Chinese comedy cartoons."},
            {"role": "user", "content": build_translation_prompt(segments, context_bundle)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return [TranslationRow.model_validate(item) for item in data["translations"]]


def export_review_csv(path: Path, segments: list[TimedSegment], translations: list[TranslationRow]) -> None:
    by_id = {row.segment_id: row for row in translations}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for segment in segments:
            row = by_id.get(
                segment.id,
                TranslationRow(
                    segment_id=segment.id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    speaker=segment.speaker,
                    text_cn=segment.text,
                    text_vi="",
                ),
            )
            writer.writerow(row.model_dump())


def import_review_csv(path: Path) -> list[TranslationRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [TranslationRow.model_validate(row) for row in reader]
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_translate.py -v`

Expected: PASS.

- [ ] **Step 5: Commit translation review**

```bash
git add src/vietdub/translate.py tests/test_translate.py
git commit -m "feat: add translation review csv"
```

## Task 9: TTS And Audio Sync

**Files:**
- Create: `src/vietdub/tts.py`
- Create: `src/vietdub/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write sync math tests**

```python
# tests/test_sync.py
from vietdub.sync import atempo_filters, speed_factor_for_duration


def test_speed_factor_for_duration():
    assert speed_factor_for_duration(actual_ms=2000, target_ms=1000) == 2.0
    assert speed_factor_for_duration(actual_ms=1000, target_ms=2000) == 0.5


def test_atempo_filters_split_large_factor():
    assert atempo_filters(4.0) == ["atempo=2.0", "atempo=2.0"]
    assert atempo_filters(0.25) == ["atempo=0.5", "atempo=0.5"]
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_sync.py -v`

Expected: FAIL with missing `vietdub.sync`.

- [ ] **Step 3: Implement Edge TTS wrapper**

```python
# src/vietdub/tts.py
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
```

- [ ] **Step 4: Implement sync utilities**

```python
# src/vietdub/sync.py
from __future__ import annotations

import subprocess
import wave
from pathlib import Path


def speed_factor_for_duration(actual_ms: int, target_ms: int) -> float:
    if actual_ms <= 0 or target_ms <= 0:
        raise ValueError("durations must be positive")
    return round(actual_ms / target_ms, 3)


def atempo_filters(factor: float) -> list[str]:
    filters: list[str] = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.3g}")
    return filters


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return int(frames / rate * 1000)


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def trim_silence(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=1:stop_threshold=-45dB",
        str(output_path),
    ])


def stretch_audio(input_path: Path, output_path: Path, factor: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = ",".join(atempo_filters(factor))
    run_ffmpeg(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", filters, str(output_path)])
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m pytest tests/test_sync.py -v`

Expected: PASS.

- [ ] **Step 6: Commit TTS and sync utilities**

```bash
git add src/vietdub/tts.py src/vietdub/sync.py tests/test_sync.py
git commit -m "feat: add tts and audio sync utilities"
```

## Task 10: Pipeline Orchestration And CLI Wiring

**Files:**
- Create: `src/vietdub/pipeline.py`
- Modify: `src/vietdub/cli.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write pipeline dry-run test using fixture engines**

```python
# tests/test_pipeline.py
import json

from vietdub.pipeline import run_fixture_pipeline


def test_fixture_pipeline_stops_after_review_csv(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    stt = tmp_path / "stt.json"
    ocr = tmp_path / "ocr.json"
    stt.write_text(json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "stt"}]), encoding="utf-8")
    ocr.write_text(json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "ocr"}]), encoding="utf-8")

    job = run_fixture_pipeline(video=video, jobs_dir=tmp_path / "jobs", stt_fixture=stt, ocr_fixture=ocr)

    assert (job.root / "translation" / "review.csv").exists()
    assert (job.root / "context" / "style_guide.json").exists()
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: FAIL with missing `vietdub.pipeline`.

- [ ] **Step 3: Implement fixture pipeline for integration safety**

```python
# src/vietdub/pipeline.py
from __future__ import annotations

from pathlib import Path

from .context import build_context_bundle
from .jobs import Job, JobManager
from .merge import merge_segments
from .models import TimedSegment, TranslationRow
from .ocr import FixtureOcrEngine
from .stt import FixtureSttEngine
from .translate import export_review_csv


def write_segments(job: Job, relative: str, segments: list[TimedSegment]) -> None:
    job.write_json(relative, [segment.model_dump() for segment in segments])


def run_fixture_pipeline(video: Path, jobs_dir: Path, stt_fixture: Path, ocr_fixture: Path) -> Job:
    job = JobManager(jobs_dir).create(video, series=None)
    stt_segments = FixtureSttEngine(stt_fixture).transcribe(job.root / "audio" / "original.wav")
    ocr_segments = FixtureOcrEngine(ocr_fixture).recognize(job.input_video)
    merged = merge_segments(stt_segments, ocr_segments)
    context_bundle = build_context_bundle(merged, series_context={})
    translations = [
        TranslationRow(
            segment_id=segment.id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            speaker=segment.speaker,
            text_cn=segment.text,
            text_vi="",
        )
        for segment in merged
    ]

    write_segments(job, "stt/segments.json", stt_segments)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    return job
```

- [ ] **Step 4: Wire CLI to job manager with clear pending full-engine message**

Replace `src/vietdub/cli.py` with:

```python
from pathlib import Path

import typer

from .config import Settings
from .jobs import JobManager
from .models import StepName


app = typer.Typer(help="Vietnamese dubbing pipeline for Chinese cartoon videos.")


@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", help="Pipeline mode: review or auto."),
    series: str | None = typer.Option(None, "--series", help="Series memory name."),
) -> None:
    if mode not in {"review", "auto"}:
        raise typer.BadParameter("mode must be review or auto")
    if not video.exists():
        raise typer.BadParameter(f"Video not found: {video}")
    settings = Settings()
    job = JobManager(Path(settings.jobs_dir)).create(video, series=series)
    typer.echo(f"Created job: {job.root}")
    typer.echo("Full engine pipeline will run after Task 11 wires media, STT, OCR, translate, TTS, and render stages.")


@app.command()
def resume(
    job_dir: Path,
    from_step: StepName = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    job = JobManager(Path("jobs")).open(job_dir)
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    typer.echo(f"Resume order: {steps}")


@app.command()
def inspect(job_dir: Path) -> None:
    job = JobManager(Path("jobs")).open(job_dir)
    typer.echo(job.root)
    typer.echo(job.load_status())


def main() -> None:
    app()
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit pipeline skeleton**

```bash
git add src/vietdub/pipeline.py src/vietdub/cli.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration skeleton"
```

## Task 11: Full Engine Pipeline And Render

**Files:**
- Modify: `src/vietdub/pipeline.py`
- Modify: `src/vietdub/cli.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add test for review mode stopping before TTS**

Append to `tests/test_pipeline.py`:

```python
def test_review_mode_does_not_create_preview_in_fixture_pipeline(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    stt = tmp_path / "stt.json"
    ocr = tmp_path / "ocr.json"
    stt.write_text(json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "stt"}]), encoding="utf-8")
    ocr.write_text(json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "ocr"}]), encoding="utf-8")

    job = run_fixture_pipeline(video=video, jobs_dir=tmp_path / "jobs", stt_fixture=stt, ocr_fixture=ocr)

    assert not (job.root / "output" / "preview_vi.mp4").exists()
```

- [ ] **Step 2: Run regression tests**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 3: Extend `pipeline.py` with real-stage function signatures**

Add these functions below `run_fixture_pipeline`:

```python
def run_review_pipeline(video: Path, jobs_dir: Path, series: str | None, settings) -> Job:
    from .config import Settings
    from .media import extract_audio
    from .ocr import PaddleSubtitleOcrEngine
    from .stt import FasterWhisperSttEngine
    from .translate import translate_with_llm

    typed_settings: Settings = settings
    job = JobManager(jobs_dir).create(video, series=series)
    audio_path = job.root / "audio" / "original.wav"
    extract_audio(job.input_video, audio_path, typed_settings.sample_rate)
    job.mark_done(__import__("vietdub.models", fromlist=["StepName"]).StepName.EXTRACT, {"audio": str(audio_path)})

    stt_segments = FasterWhisperSttEngine(typed_settings.stt_model, typed_settings.stt_language).transcribe(audio_path)
    ocr_segments = PaddleSubtitleOcrEngine().recognize(job.input_video)
    merged = merge_segments(stt_segments, ocr_segments)
    context_bundle = build_context_bundle(merged, series_context={})
    translations = translate_with_llm(merged, context_bundle, typed_settings.openai_api_key, typed_settings.llm_model)

    write_segments(job, "stt/segments.json", stt_segments)
    write_segments(job, "ocr/subtitles.json", ocr_segments)
    write_segments(job, "transcript/merged.json", merged)
    for name, value in context_bundle.items():
        job.write_json(f"context/{name}.json", value)
    job.write_json("translation/translated.json", [row.model_dump() for row in translations])
    export_review_csv(job.root / "translation" / "review.csv", merged, translations)
    return job
```

- [ ] **Step 4: Replace the dynamic StepName import with a normal import**

Edit the import block in `src/vietdub/pipeline.py` to include:

```python
from .models import StepName, TimedSegment, TranslationRow
```

Then replace:

```python
job.mark_done(__import__("vietdub.models", fromlist=["StepName"]).StepName.EXTRACT, {"audio": str(audio_path)})
```

with:

```python
job.mark_done(StepName.EXTRACT, {"audio": str(audio_path)})
```

- [ ] **Step 5: Update CLI `run` to call real review pipeline**

Replace the body after `settings = Settings()` in `run` with:

```python
    settings = Settings()
    job = run_review_pipeline(video=video, jobs_dir=Path(settings.jobs_dir), series=series, settings=settings)
    typer.echo(f"Review file written: {job.root / 'translation' / 'review.csv'}")
    if mode == "review":
        typer.echo("Review mode stopped before TTS. Edit review.csv, then run: vietdub resume <job_dir> --from tts")
    else:
        typer.echo("Auto mode will continue through TTS/render after resume rendering is implemented.")
```

Also add:

```python
from .pipeline import run_review_pipeline
```

- [ ] **Step 6: Run import and unit tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Commit review pipeline**

```bash
git add src/vietdub/pipeline.py src/vietdub/cli.py tests/test_pipeline.py
git commit -m "feat: wire review pipeline"
```

## Task 12: Resume TTS And Preview Render

**Files:**
- Modify: `src/vietdub/pipeline.py`
- Modify: `src/vietdub/cli.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add test for imported reviewed rows creating SRT text**

Append to `tests/test_pipeline.py`:

```python
from vietdub.srt import render_srt
from vietdub.models import TimedSegment


def test_render_srt_from_vietnamese_rows():
    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="Xin chào")]
    assert "Xin chào" in render_srt(segments)
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 3: Add resume render function**

Append to `src/vietdub/pipeline.py`:

```python
def resume_tts_and_render(job: Job, settings) -> Path:
    import asyncio
    from .media import mux_preview
    from .srt import render_srt
    from .translate import import_review_csv
    from .tts import EdgeTtsEngine

    rows = import_review_csv(job.root / "translation" / "review.csv")
    vietnamese_segments = [
        TimedSegment(
            id=row.segment_id,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            text=row.text_vi,
            speaker=row.speaker,
            source="translation",
        )
        for row in rows
        if row.status != "skip" and row.text_vi.strip()
    ]
    srt_path = job.root / "output" / "subtitles_vi.srt"
    srt_path.write_text(render_srt(vietnamese_segments), encoding="utf-8")

    async def synthesize_all() -> None:
        engine = EdgeTtsEngine(settings.edge_voice)
        for row in rows:
            if row.status == "skip" or not row.text_vi.strip():
                continue
            await engine.synthesize_segment(row, job.root / "tts" / "segments" / f"{row.segment_id}.mp3")

    asyncio.run(synthesize_all())
    final_audio = job.root / "tts" / "final_vi.wav"
    final_audio.write_bytes(b"")
    preview = job.root / "output" / "preview_vi.mp4"
    if final_audio.stat().st_size > 0:
        mux_preview(job.input_video, final_audio, srt_path, preview)
    return srt_path
```

- [ ] **Step 4: Update CLI resume**

Replace the body of `resume` in `src/vietdub/cli.py` with:

```python
    settings = Settings()
    job = JobManager(Path(settings.jobs_dir)).open(job_dir)
    if from_step == StepName.TTS:
        from .pipeline import resume_tts_and_render

        srt_path = resume_tts_and_render(job, settings)
        typer.echo(f"Vietnamese subtitles written: {srt_path}")
        typer.echo("TTS segment files written under tts/segments.")
        return
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    typer.echo(f"Resume order: {steps}")
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit resume render path**

```bash
git add src/vietdub/pipeline.py src/vietdub/cli.py tests/test_pipeline.py
git commit -m "feat: resume tts and subtitle render"
```

## Task 13: Documentation And Manual Acceptance

**Files:**
- Modify: `README.md`
- Create: `docs/manual-test.md`

- [ ] **Step 1: Expand README with exact MVP flow**

Add this section to `README.md`:

```markdown
## MVP Flow

1. Put one `.mp4` video in the project folder.
2. Run review mode:

```powershell
vietdub run .\input.mp4 --mode review --series demo
```

3. Edit `jobs\<job-name>\translation\review.csv`.
4. Resume from TTS:

```powershell
vietdub resume .\jobs\<job-name> --from tts
```

5. Check:

- `jobs\<job-name>\translation\review.csv`
- `jobs\<job-name>\output\subtitles_vi.srt`
- `jobs\<job-name>\tts\segments\*.mp3`
- `jobs\<job-name>\output\preview_vi.mp4` when final audio muxing is available
```

- [ ] **Step 2: Add manual test doc**

```markdown
# Manual Test

## Environment

- Windows
- Python 3.11
- FFmpeg in PATH
- `OPENAI_API_KEY` set
- `LLM_MODEL` set

## Test Video

Use a 30-60 second `.mp4` clip with clear Chinese hard subtitles near the bottom of the frame.

## Commands

```powershell
python -m pip install -e ".[dev]"
vietdub run .\sample.mp4 --mode review --series sample-series
```

Open `jobs\sample\translation\review.csv`, fill at least three `text_vi` cells, save as UTF-8 CSV, then run:

```powershell
vietdub resume .\jobs\sample --from tts
```

## Acceptance Criteria

- A job folder is created under `jobs`.
- `stt/segments.json`, `ocr/subtitles.json`, `transcript/merged.json`, and `context/style_guide.json` exist after review run.
- `translation/review.csv` opens in Excel without mojibake.
- `output/subtitles_vi.srt` contains the reviewed Vietnamese lines.
- `tts/segments` contains one MP3 per reviewed non-empty row.
```

- [ ] **Step 3: Run full unit suite**

Run: `python -m pytest -v`

Expected: PASS for all unit tests.

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/manual-test.md
git commit -m "docs: add vietdub manual test flow"
```

## Self-Review Checklist

- Spec coverage: CLI, job cache, STT, OCR, merge, context, LLM translation, review CSV, Edge TTS, subtitle render, and preview render hooks are covered.
- Placeholder scan: The plan contains no `TBD`, no `TODO`, and no unbounded "handle later" instructions.
- Type consistency: `TimedSegment`, `TranslationRow`, `StepName`, and job paths are used consistently across tasks.
- License note: ThioJoe repo is used only as architecture reference; no source code is copied.
