from pathlib import Path

import typer


app = typer.Typer(help="Vietnamese dubbing pipeline for Chinese cartoon videos.")


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
    typer.echo(f"Prepared to run {video} in {mode} mode")


@app.command()
def resume(
    job_dir: Path,
    from_step: str = typer.Option(..., "--from", help="Step to resume from."),
) -> None:
    if not job_dir.exists():
        raise typer.BadParameter(f"Job directory not found: {job_dir}")
    typer.echo(f"Prepared to resume {job_dir} from {from_step}")


@app.command()
def inspect(job_dir: Path) -> None:
    if not job_dir.exists():
        raise typer.BadParameter(f"Job directory not found: {job_dir}")
    typer.echo(job_dir)


def main() -> None:
    app()
