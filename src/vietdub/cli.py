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


def _resolve_job_path(
    job_dir: Path | None,
    job_name: str | None,
    jobs_dir: Path,
) -> Path:
    """Resolve a CLI job reference to an absolute path.

    - If job_dir is given and absolute, use as-is.
    - If job_dir is given and exists, use as-is.
    - If job_dir is given but doesn't exist as a literal, try under jobs_dir.
    - If job_name is given, use jobs_dir / job_name.
    - Otherwise raise.
    """
    if job_dir is not None:
        if job_dir.is_absolute() or job_dir.exists():
            return job_dir
        return jobs_dir / job_dir
    if job_name is not None:
        return jobs_dir / job_name
    raise typer.BadParameter("Either job_dir or --job-name is required")


def _open_job(target: Path, jobs_dir: Path) -> Job:
    try:
        return JobManager(jobs_dir).open(target)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def run(
    video: Path,
    mode: str = typer.Option("review", "--mode", help="Pipeline mode: review or auto."),
    series: str | None = typer.Option(None, "--series", help="Series memory name."),
    no_review: bool = typer.Option(False, "--no-review", help="Skip length review pass (saves LLM cost)."),
) -> None:
    if mode not in {"review", "auto"}:
        raise typer.BadParameter("mode must be review or auto")
    if not video.exists():
        raise typer.BadParameter(f"Video not found: {video}")
    settings = Settings()
    try:
        job = run_review_pipeline(
            video=video,
            jobs_dir=Path(settings.jobs_dir),
            series=series,
            settings=settings,
            review=not no_review,
        )
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
    job_dir: Path = typer.Argument(None, help="Path to job dir, or a bare name to resolve under JOBS_DIR."),
    job_name: str | None = typer.Option(None, "--job-name", help="Bare job name under JOBS_DIR."),
    from_step: StepName = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    settings = Settings()
    target = _resolve_job_path(job_dir, job_name, Path(settings.jobs_dir))
    job = _open_job(target, Path(settings.jobs_dir))
    if from_step == StepName.TTS:
        try:
            srt_path = resume_tts_and_render(job, settings)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None
        safe_echo(f"Vietnamese subtitles written: {srt_path}")
        safe_echo("TTS segment files written under tts/segments.")
        return
    raise click.ClickException("Only resume from tts is currently supported.")


@app.command()
def inspect(
    job_dir: Path = typer.Argument(None, help="Path to job dir, or a bare name to resolve under JOBS_DIR."),
    job_name: str | None = typer.Option(None, "--job-name", help="Bare job name under JOBS_DIR."),
) -> None:
    settings = Settings()
    target = _resolve_job_path(job_dir, job_name, Path(settings.jobs_dir))
    job = _open_job(target, Path(settings.jobs_dir))
    safe_echo(job.root)
    safe_echo(job.load_status())


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    app()
