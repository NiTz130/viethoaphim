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
