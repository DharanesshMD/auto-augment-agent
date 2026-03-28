"""CLI entry point for the `ada` command."""

from __future__ import annotations

import click

from src.utils.config import PROJECT_ROOT


@click.group()
@click.version_option(package_name="auto-augment-agent")
def main():
    """🧬 ADA — Autonomous Data-Augmentation & Tuning Agent.

    Iteratively discovers augmentation pipelines and hyperparameters
    that improve your model's validation metrics.
    """
    pass


@main.command()
@click.option(
    "--preset",
    type=click.Choice(["nlp-quick", "cv-quick", "full"]),
    default=None,
    help="Use a preset configuration instead of the interactive wizard.",
)
def init(preset: str | None):
    """🚀 Initialize project configuration via interactive setup wizard."""
    from scripts.init import run_setup_wizard

    run_setup_wizard(preset=preset)


@main.command()
@click.option("--max-trials", "-n", default=50, help="Maximum number of trials to run.")
@click.option("--resume/--no-resume", default=False, help="Resume from last checkpoint.")
@click.option("--docker/--no-docker", default=False, help="Run trials in Docker containers.")
@click.option("--dry-run", is_flag=True, help="Propose configs without executing trials.")
def run(max_trials: int, resume: bool, docker: bool, dry_run: bool):
    """🤖 Start the autonomous agent loop."""
    from scripts.run_agent import run_agent

    run_agent(
        max_trials=max_trials,
        resume=resume,
        use_docker=docker,
        dry_run=dry_run,
    )


@main.command()
@click.option("--device", default="auto", help="Device to use (auto/cuda/mps/cpu).")
@click.option("--max-steps", default=500, help="Training steps for baseline.")
def baseline(device: str, max_steps: int):
    """📊 Establish baseline metrics with default configuration."""
    from scripts.create_baseline import create_baseline

    create_baseline(device=device, max_steps=max_steps)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input", "-i", "input_text", default=None, help="Input string for the model.")
@click.option("--device", default="auto", help="Device to use.")
def test(model_path: str, input_text: str | None, device: str):
    """🔮 Run inference on a trained model or trial."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/test_model.py", "--model-path", model_path, "--device", device]
    if input_text:
        cmd.extend(["--input", input_text])
    
    subprocess.run(cmd)


@main.command()
@click.option("--output", "-o", default=None, help="Output path for the report.")
def analyze(output: str | None):
    """📋 Analyze completed trial results and generate report."""
    from scripts.analyze_results import analyze

    analyze(output_path=output)


@main.command()
def info():
    """ℹ️  Show current configuration and system info."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from src.utils.config import load_config
    from src.utils.reproducibility import get_device, get_device_info, get_system_info

    console = Console()

    # System info
    sys_info = get_system_info()
    device = get_device("auto")
    dev_info = get_device_info(device)

    table = Table(title="System Information", show_header=False, border_style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for k, v in {**sys_info, **dev_info}.items():
        table.add_row(k, str(v))
    console.print(table)

    # Config summary
    try:
        config = load_config()
        console.print(Panel(
            f"[bold]Task:[/bold] {config.get('model', {}).get('task', 'N/A')}\n"
            f"[bold]Model:[/bold] {config.get('model', {}).get('name', 'N/A')}\n"
            f"[bold]Dataset:[/bold] {config.get('dataset', {}).get('name', 'N/A')}\n"
            f"[bold]LLM:[/bold] {config.get('llm', {}).get('provider', 'N/A')} / "
            f"{config.get('llm', {}).get('model', 'N/A')}\n"
            f"[bold]Tracker:[/bold] {config.get('tracking', {}).get('backend', 'none')}\n"
            f"[bold]Execution:[/bold] {config.get('execution', {}).get('mode', 'local')}",
            title="Configuration",
            border_style="green",
        ))
    except FileNotFoundError:
        console.print("[yellow]⚠ No configuration found. Run `ada init` first.[/yellow]")


if __name__ == "__main__":
    main()
