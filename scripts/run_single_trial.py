#!/usr/bin/env python3
"""Run a single training trial — used inside Docker containers.

Usage:
    python scripts/run_single_trial.py --trial-id 0
    python scripts/run_single_trial.py --trial-id 5 --device cuda --max-steps 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_trial(
    trial_id: int,
    device: str = "auto",
    max_steps: int | None = None,
    config_dir: str | None = None,
    output_dir: str | None = None,
):
    """Run a single training trial."""
    from src.data.loader import create_dataloaders, load_dataset_from_config
    from src.training.augmentations import AugmentationPipeline
    from src.training.train import Trainer
    from src.utils.config import CONFIG_DIR, load_config, load_yaml
    from src.utils.reproducibility import set_seed

    # Load config
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    config = load_config()

    if max_steps:
        config["training"]["max_steps"] = max_steps
    if device != "auto":
        config["training"]["device"] = device

    set_seed(config.get("project", {}).get("seed", 42) + trial_id)

    # Load augmentation config
    aug_config = load_yaml(cfg_dir / "augmentations.yaml")
    aug_pipeline = AugmentationPipeline(aug_config)

    # Build trainer
    trainer = Trainer(config)
    model = trainer.load_model()

    # Apply LoRA if enabled
    if config.get("lora", {}).get("enabled", False):
        from src.training.lora import apply_lora
        model = apply_lora(model, config["lora"])

    # Load data
    splits = load_dataset_from_config(config)
    loaders = create_dataloaders(splits, config)

    # Train
    results = trainer.train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders.get("validation"),
        augmentation_fn=aug_pipeline.augment_text if aug_pipeline.enabled else None,
    )

    # Save results
    out_dir = Path(output_dir) if output_dir else Path("outputs") / f"trial_{trial_id}"
    results_path = trainer.save_results(results, out_dir)

    print(f"Trial {trial_id} complete. Results: {results_path}")
    print(json.dumps(results.get("eval_metrics", {}), indent=2))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a single training trial")
    parser.add_argument("--trial-id", type=int, required=True, help="Trial identifier")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--config-dir", default=None, help="Config directory path")
    parser.add_argument("--output-dir", default=None, help="Output directory path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_trial(
        trial_id=args.trial_id,
        device=args.device,
        max_steps=args.max_steps,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
