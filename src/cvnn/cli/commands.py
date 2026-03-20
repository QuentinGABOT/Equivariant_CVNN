"""
cvnn.cli.commands

Defines the CLI entry point and subcommands for CVNN experiments.
"""

# Standard library imports
import logging
from typing import Optional

# Third-party imports
import click

# Local imports
import cvnn.cli as cli_pkg

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["full", "train", "retrain", "eval"]),
    default=None,
    help="Pipeline mode override (full/train/retrain/eval).",
)
@click.option(
    "--resume-logdir",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Existing log directory to resume from for retrain/eval.",
)
def cli_main(
    config_path: str,
    mode: Optional[str],
    resume_logdir: Optional[str],
) -> None:
    """Run a CVNN experiment based on the provided YAML config file."""
    click.echo(f"Running experiment with config: {config_path}")
    if mode is not None:
        click.echo(f"Mode override: {mode}")
        logger.info(f"Mode override: {mode}")
    if resume_logdir is not None:
        click.echo(f"Resuming from log directory: {resume_logdir}")
        logger.info(f"Resuming from log directory: {resume_logdir}")

    # Delegate to run_experiment dynamically to allow monkeypatching via cvnn.cli
    cli_pkg.run_experiment(config_path, resume_logdir=resume_logdir, mode_override=mode)
