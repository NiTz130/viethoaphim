from pathlib import Path

import pytest

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


def test_create_job_appends_suffix_when_directory_already_exists(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    manager = JobManager(tmp_path / "jobs")

    first = manager.create(video, series=None)
    second = manager.create(video, series=None)
    third = manager.create(video, series=None)

    assert first.root.name == "clip"
    assert second.root.name == "clip-1"
    assert third.root.name == "clip-2"
    assert (second.root / "input.mp4").read_bytes() == b"fake-video"


def test_create_job_skips_taken_suffix_until_free_slot_found(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    manager = JobManager(tmp_path / "jobs")

    manager.create(video, series=None)
    (manager.jobs_dir / "clip-1").mkdir(parents=True)
    (manager.jobs_dir / "clip-2").mkdir(parents=True)

    job = manager.create(video, series=None)

    assert job.root.name == "clip-3"


def test_mark_step_and_inspect(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job = JobManager(tmp_path / "jobs").create(video, series=None)

    job.mark_done(StepName.STT, {"segments": 2})
    status = job.load_status()

    assert status["stt"]["state"] == "done"
    assert status["stt"]["details"] == {"segments": 2}


def test_create_job_rejects_oversized_video(tmp_path, monkeypatch):
    """L3: JobManager.create raises RuntimeError when the video exceeds max_video_size_mb."""
    from vietdub.jobs import JobManager

    video = tmp_path / "big.mp4"
    # Write 2 MB of zeros.
    video.write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setenv("MAX_VIDEO_SIZE_MB", "1")

    manager = JobManager(tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="Video too large"):
        manager.create(video, series=None)


def test_create_job_accepts_oversized_video_when_limit_is_zero(tmp_path, monkeypatch):
    """L3: max_video_size_mb=0 disables the check."""
    from vietdub.jobs import JobManager

    video = tmp_path / "big.mp4"
    video.write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setenv("MAX_VIDEO_SIZE_MB", "0")

    manager = JobManager(tmp_path / "jobs")
    job = manager.create(video, series=None)
    assert (job.root / "input.mp4").read_bytes() == b"x" * (2 * 1024 * 1024)
