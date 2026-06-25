from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

from .models import TimedSegment


class FixtureOcrEngine:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path

    def recognize(self, video_path: Path) -> list[TimedSegment]:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return [TimedSegment.model_validate(item) for item in data]


def _clear_generated_ocr_frames(frame_dir: Path) -> None:
    if not frame_dir.exists():
        return
    for frame in frame_dir.glob("frame_*.jpg"):
        if frame.is_file():
            frame.unlink()


class PaddleSubtitleOcrEngine:
    def __init__(self, sample_every_seconds: float = 0.5) -> None:
        self.sample_every_seconds = sample_every_seconds

    def recognize(self, video_path: Path) -> list[TimedSegment]:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        _guard_optional_torch_import()
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=False, enable_mkldnn=False)
        frame_dir = video_path.parent / "ocr_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _clear_generated_ocr_frames(frame_dir)
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


def _guard_optional_torch_import() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    except OSError as exc:
        message = str(exc).lower()
        if "torch" not in message and "shm.dll" not in message:
            raise
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object

        def _missing_torch(*args, **kwargs):
            raise RuntimeError("PyTorch tensor transforms are unavailable in this environment")

        torch_stub.from_numpy = _missing_torch
        sys.modules["torch"] = torch_stub
