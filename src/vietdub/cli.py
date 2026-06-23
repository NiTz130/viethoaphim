from pathlib import Path

import typer

from .config import Settings
from .jobs import Job, JobManager
from .models import StepName


app = typer.Typer(help="Vietnamese dubbing pipeline for Chinese cartoon videos.")


def _open_job(job_dir: Path, jobs_dir: Path) -> Job:
    try:
        return JobManager(jobs_dir).open(job_dir)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    job = _open_job(job_dir, Path("jobs"))
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    typer.echo(f"Resume order: {steps}")


@app.command()
def inspect(job_dir: Path) -> None:
    job = _open_job(job_dir, Path("jobs"))
    typer.echo(job.root)
    typer.echo(job.load_status())


def main() -> None:
    app()
