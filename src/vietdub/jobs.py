from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StepName


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


class JobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir

    def create(self, video: Path, series: str | None) -> Job:
        root = self.jobs_dir / video.stem
        suffix = 1
        while True:
            try:
                root.mkdir(parents=True)
                break
            except FileExistsError:
                root = self.jobs_dir / f"{video.stem}-{suffix}"
                suffix += 1
        for relative in JOB_DIRS:
            (root / relative).mkdir(parents=True, exist_ok=True)
        # Check video size before copying (L3).
        size_bytes = video.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        limit_mb = 10240
        try:
            from .config import Settings
            limit_mb = Settings().max_video_size_mb
        except Exception:
            pass
        if limit_mb > 0 and size_mb > limit_mb:
            raise RuntimeError(
                f"Video too large: {size_mb:.1f}MB exceeds max_video_size_mb={limit_mb}MB. "
                f"Set MAX_VIDEO_SIZE_MB=0 in env (or jobs.max_video_size_mb=0 in .env) to disable."
            )

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
