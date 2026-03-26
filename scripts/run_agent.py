#!/usr/bin/env python3
"""Entry point: Start the autonomous augmentation agent loop.

Usage:
    python scripts/run_agent.py --max-trials 50 --no-docker
    python scripts/run_agent.py --max-trials 10 --docker --resume
    python scripts/run_agent.py --dry-run --max-trials 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def run_agent(
    max_trials: int = 50,
    resume: bool = False,
    use_docker: bool = False,
    dry_run: bool = False,
):
    """Run the autonomous agent."""
    from dotenv import load_dotenv

    from src.agent.runner import AgentRunner
    from src.utils.config import load_config

    load_dotenv()
    config = load_config()

    runner = AgentRunner(
        config=config,
        use_docker=use_docker,
        dry_run=dry_run,
    )

    decisions = runner.run(
        max_trials=max_trials,
        resume=resume,
    )

    # Exit with success if at least one trial was accepted
    accepted = sum(1 for d in decisions if d.accepted)
    if accepted > 0:
        console.print(f"\n[green]✅ {accepted} trial(s) accepted![/green]")
    else:
        console.print("\n[yellow]⚠ No trials accepted.[/yellow]")

    return decisions


def main():
    parser = argparse.ArgumentParser(
        description="🧬 Auto-Augment Agent — Autonomous Data-Augmentation & Tuning"
    )
    parser.add_argument(
        "--max-trials", "-n",
        type=int,
        default=50,
        help="Maximum number of trials to run (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved checkpoint",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        dest="docker",
        help="Run trials in Docker containers",
    )
    parser.add_argument(
        "--no-docker",
        action="store_false",
        dest="docker",
        help="Run trials locally (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Propose configs without executing training",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(docker=False)

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    run_agent(
        max_trials=args.max_trials,
        resume=args.resume,
        use_docker=args.docker,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
