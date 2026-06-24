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
