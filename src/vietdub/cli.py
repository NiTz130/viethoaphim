from pathlib import Path

import click
import typer

from .config import Settings
from .jobs import Job, JobManager
from .models import StepName
from .pipeline import run_review_pipeline


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
    try:
        job = run_review_pipeline(video=video, jobs_dir=Path(settings.jobs_dir), series=series, settings=settings)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from None
    typer.echo(f"Review file written: {job.root / 'translation' / 'review.csv'}")
    if mode == "review":
        typer.echo("Review mode stopped before TTS. Edit review.csv, then run: vietdub resume <job_dir> --from tts")
    else:
        typer.echo("Auto mode will continue through TTS/render after resume rendering is implemented.")


@app.command()
def resume(
    job_dir: Path,
    from_step: StepName = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    settings = Settings()
    job = _open_job(job_dir, Path(settings.jobs_dir))
    if from_step == StepName.TTS:
        from .pipeline import resume_tts_and_render

        srt_path = resume_tts_and_render(job, settings)
        typer.echo(f"Vietnamese subtitles written: {srt_path}")
        typer.echo("TTS segment files written under tts/segments.")
        return
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    typer.echo(f"Resume order: {steps}")


@app.command()
def inspect(job_dir: Path) -> None:
    job = _open_job(job_dir, Path("jobs"))
    typer.echo(job.root)
    typer.echo(job.load_status())


def main() -> None:
    app()
