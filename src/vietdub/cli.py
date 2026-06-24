from pathlib import Path

import click
import typer

from .config import Settings
from .jobs import Job, JobManager
from .models import StepName
from .pipeline import resume_tts_and_render, run_review_pipeline


app = typer.Typer(help="Vietnamese dubbing pipeline for Chinese cartoon videos.")


def safe_echo(message: object) -> None:
    text = str(message)
    typer.echo(text.encode("ascii", errors="replace").decode("ascii"))


def _open_job(job_dir: Path, jobs_dir: Path) -> Job:
    candidate = jobs_dir / job_dir if not job_dir.is_absolute() and len(job_dir.parts) == 1 else job_dir
    try:
        return JobManager(jobs_dir).open(candidate)
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
    safe_echo(f"Review file written: {job.root / 'translation' / 'review.csv'}")
    if mode == "review":
        safe_echo("Review mode stopped before TTS. Edit review.csv, then run: vietdub resume <job_dir> --from tts")
    else:
        try:
            srt_path = resume_tts_and_render(job, settings)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None
        safe_echo(f"Vietnamese subtitles written: {srt_path}")
        safe_echo("TTS segment files written under tts/segments.")


@app.command()
def resume(
    job_dir: Path,
    from_step: StepName = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    settings = Settings()
    job = _open_job(job_dir, Path(settings.jobs_dir))
    if from_step == StepName.TTS:
        try:
            srt_path = resume_tts_and_render(job, settings)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None
        safe_echo(f"Vietnamese subtitles written: {srt_path}")
        safe_echo("TTS segment files written under tts/segments.")
        return
    steps = ", ".join(step.value for step in job.steps_from(from_step))
    safe_echo(f"Resume order: {steps}")


@app.command()
def inspect(job_dir: Path) -> None:
    settings = Settings()
    job = _open_job(job_dir, Path(settings.jobs_dir))
    safe_echo(job.root)
    safe_echo(job.load_status())


def main() -> None:
    app()
