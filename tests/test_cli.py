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
