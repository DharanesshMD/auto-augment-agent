#!/usr/bin/env python3
"""Analyze completed trial results and generate a summary report.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --output outputs/report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def analyze(output_path: str | None = None):
    """Analyze trial results from the provenance database."""
    from src.tracking.provenance import TrialDatabase
    from src.utils.config import PROJECT_ROOT

    db_path = PROJECT_ROOT / "provenance" / "trials.db"
    if not db_path.exists():
        console.print("[red]No trial database found. Run some trials first.[/red]")
        return

    db = TrialDatabase(str(db_path))
    stats = db.get_stats()
    all_trials = db.get_all()
    accepted = db.get_accepted()

    # Summary table
    summary = Table(title="🧬 Trial Analysis Summary", border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", style="cyan")

    summary.add_row("Total trials", str(stats["total_trials"]))
    summary.add_row("Accepted", f"[green]{stats['accepted']}[/green]")
    summary.add_row("Rejected", f"[red]{stats['rejected']}[/red]")
    summary.add_row(
        "Acceptance rate",
        f"{stats['acceptance_rate'] * 100:.1f}%",
    )

    console.print(summary)

    # Trial details table
    if all_trials:
        detail_table = Table(
            title="Trial History",
            border_style="dim",
            show_lines=True,
        )
        detail_table.add_column("#", style="dim", width=4)
        detail_table.add_column("Decision", width=10)
        detail_table.add_column("Primary Metric", width=15)
        detail_table.add_column("Reason", max_width=50)

        for trial in all_trials[-20:]:  # Show last 20
            decision_style = "green" if trial["decision"] == "accepted" else "red"
            metric = trial.get("primary_metric")
            metric_str = f"{metric:.4f}" if metric is not None else "N/A"

            detail_table.add_row(
                str(trial["trial_id"]),
                f"[{decision_style}]{trial['decision']}[/{decision_style}]",
                metric_str,
                (trial.get("reason") or "")[:50],
            )

        console.print(detail_table)

    # Best trial
    best = db.get_best()
    if best:
        console.print(Panel(
            f"[bold]Trial ID:[/bold] {best['trial_id']}\n"
            f"[bold]Primary Metric:[/bold] {best.get('primary_metric', 'N/A')}\n"
            f"[bold]Config Hash:[/bold] {best.get('config_hash', 'N/A')}\n"
            f"[bold]Reason:[/bold] {best.get('reason', 'N/A')}",
            title="🏆 Best Accepted Trial",
            border_style="green",
        ))

    # Generate markdown report
    if output_path:
        report = _generate_markdown_report(stats, all_trials, accepted, best)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        console.print(f"\n[green]Report saved to: {output_file}[/green]")

    db.close()


def _generate_markdown_report(
    stats: dict,
    all_trials: list,
    accepted: list,
    best: dict | None,
) -> str:
    """Generate a markdown report."""
    lines = [
        "# 🧬 Auto-Augment Agent — Trial Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total trials | {stats['total_trials']} |",
        f"| Accepted | {stats['accepted']} |",
        f"| Rejected | {stats['rejected']} |",
        f"| Acceptance rate | {stats['acceptance_rate'] * 100:.1f}% |",
        "",
    ]

    if best:
        lines.extend([
            "## Best Trial",
            "",
            f"- **Trial ID:** {best['trial_id']}",
            f"- **Primary Metric:** {best.get('primary_metric', 'N/A')}",
            f"- **Config Hash:** {best.get('config_hash', 'N/A')}",
            f"- **Reason:** {best.get('reason', 'N/A')}",
            "",
        ])

    lines.extend([
        "## Trial History",
        "",
        "| Trial | Decision | Primary Metric | Reason |",
        "|-------|----------|----------------|--------|",
    ])

    for trial in all_trials:
        metric = trial.get("primary_metric")
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        decision = "✅" if trial["decision"] == "accepted" else "❌"
        reason = (trial.get("reason") or "")[:60]
        lines.append(f"| {trial['trial_id']} | {decision} | {metric_str} | {reason} |")

    if accepted:
        lines.extend([
            "",
            "## Accepted Config Changes",
            "",
        ])
        for trial in accepted:
            diff = trial.get("diff_json", "[]")
            try:
                diff_lines = json.loads(diff) if isinstance(diff, str) else diff
            except json.JSONDecodeError:
                diff_lines = []

            lines.append(f"### Trial {trial['trial_id']}")
            lines.append("```")
            for d in diff_lines:
                lines.append(f"  {d}")
            lines.append("```")
            lines.append("")

    lines.extend([
        "",
        "---",
        "*Report generated by Auto-Augment Agent*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze trial results")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for markdown report",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    analyze(output_path=args.output)


if __name__ == "__main__":
    main()
