#!/usr/bin/env python3
"""Establish baseline metrics with default configuration (no augmentations).

Usage:
    python scripts/create_baseline.py
    python scripts/create_baseline.py --device cuda --max-steps 500
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

console = Console()


def create_baseline(device: str = "auto", max_steps: int = 500):
    """Train with default config to establish baseline metrics."""
    from src.data.loader import create_dataloaders, load_dataset_from_config
    from src.training.train import Trainer
    from src.utils.config import PROJECT_ROOT, load_config
    from src.utils.reproducibility import set_seed

    config = load_config()
    config["training"]["device"] = device
    config["training"]["max_steps"] = max_steps

    set_seed(config.get("project", {}).get("seed", 42))

    console.print(Panel(
        f"[bold]Task:[/bold] {config['model']['task']}\n"
        f"[bold]Model:[/bold] {config['model']['name']}\n"
        f"[bold]Dataset:[/bold] {config['dataset']['name']}\n"
        f"[bold]Steps:[/bold] {max_steps}\n"
        f"[bold]Device:[/bold] {device}",
        title="📊 Establishing Baseline",
        border_style="cyan",
    ))

    # Train without augmentations
    trainer = Trainer(config)
    model = trainer.load_model()

    splits = load_dataset_from_config(config)
    loaders = create_dataloaders(splits, config)

    results = trainer.train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders.get("validation"),
    )

    # Save baseline
    output_dir = PROJECT_ROOT / "outputs" / "baseline"
    results_path = trainer.save_results(results, output_dir)
    
    # Save model if requested
    if config.get("training", {}).get("save_checkpoints", False):
        model_path = trainer.save_model(model, output_dir)
        console.print(f"[dim]Model saved to: {model_path}[/dim]")

    # Also save to a known location for the agent
    baseline_path = PROJECT_ROOT / "outputs" / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump({
            "metrics": results.get("eval_metrics", {}),
            "primary_metric": results.get("primary_metric_value"),
            "steps": max_steps,
            "config": {
                "model": config["model"]["name"],
                "task": config["model"]["task"],
                "dataset": config["dataset"]["name"],
            },
        }, f, indent=2)

    # Display results
    eval_metrics = results.get("eval_metrics", {})
    metrics_str = "\n".join(f"  [bold]{k}:[/bold] {v:.4f}" for k, v in eval_metrics.items())

    console.print(Panel(
        f"[green]✅ Baseline established[/green]\n\n{metrics_str}\n\n"
        f"[dim]Saved to: {results_path}[/dim]",
        title="Baseline Results",
        border_style="green",
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Establish baseline metrics")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps for baseline")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    create_baseline(device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
