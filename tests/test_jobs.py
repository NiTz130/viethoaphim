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
