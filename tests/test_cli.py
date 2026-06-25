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


def test_safe_echo_handles_non_ascii_path(monkeypatch):
    from vietdub import cli

    captured = []
    monkeypatch.setattr(cli.typer, "echo", captured.append)

    cli.safe_echo("jobs\\T\u1eadp 1\\translation\\review.csv")

    assert captured == ["jobs\\T?p 1\\translation\\review.csv"]


def test_run_auto_resumes_tts_and_render(monkeypatch, tmp_path):
    from vietdub import cli
    from vietdub.jobs import Job

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = Job(root=job_root, config={})
    calls = []

    def fake_run_review_pipeline(video, jobs_dir, series, settings):
        calls.append(("review", video, jobs_dir, series))
        return job

    def fake_resume_tts_and_render(job_arg, settings):
        calls.append(("resume", job_arg))
        return job_arg.root / "output" / "subtitles_vi.srt"

    monkeypatch.setattr(cli, "run_review_pipeline", fake_run_review_pipeline)
    monkeypatch.setattr(cli, "resume_tts_and_render", fake_resume_tts_and_render)

    result = runner.invoke(app, ["run", str(video), "--mode", "auto"])

    assert result.exit_code == 0
    assert calls[0][0] == "review"
    assert calls[1] == ("resume", job)
    assert "Vietnamese subtitles written" in result.output


def test_run_auto_runtime_error_is_click_error(monkeypatch, tmp_path):
    from vietdub import cli
    from vietdub.jobs import Job

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = Job(root=job_root, config={})

    def fake_run_review_pipeline(video, jobs_dir, series, settings):
        return job

    def fail_resume(job_arg, settings):
        raise RuntimeError("bad tts")

    monkeypatch.setattr(cli, "run_review_pipeline", fake_run_review_pipeline)
    monkeypatch.setattr(cli, "resume_tts_and_render", fail_resume)

    result = runner.invoke(app, ["run", str(video), "--mode", "auto"])

    assert result.exit_code != 0
    assert "bad tts" in result.output
    assert "Traceback" not in result.output


def test_resume_runtime_error_is_click_error(monkeypatch, tmp_path):
    from vietdub import cli
    from vietdub.jobs import Job

    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = Job(root=job_root, config={})

    monkeypatch.setattr(cli, "_open_job", lambda job_dir, jobs_dir: job)

    def fail_resume(job_arg, settings):
        raise RuntimeError("bad review csv")

    monkeypatch.setattr(cli, "resume_tts_and_render", fail_resume)

    result = runner.invoke(app, ["resume", str(job_root), "--from", "tts"])

    assert result.exit_code != 0
    assert "bad review csv" in result.output
    assert "Traceback" not in result.output


def test_resume_rejects_unsupported_from_step(monkeypatch, tmp_path):
    from vietdub.jobs import Job

    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = Job(root=job_root, config={})

    monkeypatch.setattr("vietdub.cli._open_job", lambda job_dir, jobs_dir: job)

    result = runner.invoke(app, ["resume", str(job_root), "--from", "stt"])

    assert result.exit_code != 0
    assert "Only resume from tts is currently supported" in result.output


def test_inspect_opens_bare_job_name_from_configured_jobs_dir(monkeypatch, tmp_path):
    jobs_dir = tmp_path / "configured-jobs"
    job_root = jobs_dir / "clip"
    job_root.mkdir(parents=True)
    (job_root / "job.json").write_text('{"series": null}', encoding="utf-8")
    (job_root / "status.json").write_text('{"extract": {"state": "done"}}', encoding="utf-8")

    monkeypatch.setenv("JOBS_DIR", str(jobs_dir))

    result = runner.invoke(app, ["inspect", "clip"])

    assert result.exit_code == 0
    assert str(job_root) in result.output
    assert "extract" in result.output


def test_inspect_bare_job_name_uses_configured_jobs_dir_over_local_dir(monkeypatch, tmp_path):
    jobs_dir = tmp_path / "configured-jobs"
    job_root = jobs_dir / "clip"
    job_root.mkdir(parents=True)
    (job_root / "job.json").write_text('{"series": null}', encoding="utf-8")
    (job_root / "status.json").write_text('{"extract": {"state": "done"}}', encoding="utf-8")
    local_clip = tmp_path / "clip"
    local_clip.mkdir()
    (local_clip / "job.json").write_text('{"series": "local"}', encoding="utf-8")
    (local_clip / "status.json").write_text('{"extract": {"state": "local"}}', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JOBS_DIR", str(jobs_dir))

    result = runner.invoke(app, ["inspect", "clip"])

    assert result.exit_code == 0
    assert str(job_root) in result.output
    assert str(local_clip) not in result.output
    assert "done" in result.output
