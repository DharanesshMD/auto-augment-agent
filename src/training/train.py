"""Core training loop with augmentation hooks and optional LoRA.

Supports language modeling, text classification, and image classification tasks.
Logs to experiment tracker and returns structured results.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.metrics import MetricTracker, compute_accuracy, compute_bpb, compute_perplexity

logger = logging.getLogger(__name__)


class Trainer:
    """Unified training loop for all supported tasks.

    Handles:
    - Model loading (HuggingFace transformers or torchvision)
    - Augmentation pipeline application
    - Standard fine-tuning and LoRA mode
    - Step-budget training (not epoch-based)
    - Evaluation with configurable metrics
    - Experiment tracking integration
    """

    def __init__(self, config: dict[str, Any], tracker=None):
        self.config = config
        self.tracker = tracker
        self.task = config["model"]["task"]
        self.device = self._resolve_device()
        self.metric_tracker = MetricTracker()

        # Set metric direction
        metric = config.get("agent", {}).get("metric", "eval_loss")
        direction = config.get("agent", {}).get("metric_direction", "minimize")
        self.primary_metric = metric
        self.metric_tracker.set_direction(metric, direction)

    def _resolve_device(self) -> torch.device:
        """Resolve the training device."""
        from src.utils.reproducibility import get_device

        return get_device(self.config.get("training", {}).get("device", "auto"))

    def load_model(self) -> nn.Module:
        """Load model based on task type."""
        model_config = self.config["model"]
        model_name = model_config["name"]

        if self.task in ("language_modeling", "text_classification"):
            return self._load_transformers_model(model_name)
        elif self.task == "image_classification":
            return self._load_vision_model(model_name)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _load_transformers_model(self, model_name: str) -> nn.Module:
        """Load a HuggingFace transformers model."""
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        if self.task == "language_modeling":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif self.task == "text_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification (SST-2)
            )
        else:
            raise ValueError(f"Unsupported NLP task: {self.task}")

        logger.info(f"Loaded model: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")
        return model.to(self.device)

    def _load_vision_model(self, model_name: str) -> nn.Module:
        """Load a torchvision model."""
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision required. Install with: pip install 'auto-augment-agent[cv]'")

        ds_name = self.config["dataset"]["name"].lower()
        num_classes = 100 if ds_name == "cifar100" else 10

        if model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported vision model: {model_name}")

        logger.info(f"Loaded model: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")
        return model.to(self.device)

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.get("optimizer", {})
        name = opt_config.get("name", "adamw")
        lr = opt_config.get("learning_rate", 5e-5)
        wd = opt_config.get("weight_decay", 0.01)

        if name == "adamw":
            betas = tuple(opt_config.get("betas", [0.9, 0.999]))
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=opt_config.get("eps", 1e-8)
            )
        elif name == "adam":
            betas = tuple(opt_config.get("betas", [0.9, 0.999]))
            return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=opt_config.get("eps", 1e-8))
        elif name == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ):
        """Create learning rate scheduler from config."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            CosineAnnealingWarmRestarts,
            LinearLR,
            SequentialLR,
        )

        sched_config = self.config.get("scheduler", {})
        name = sched_config.get("name", "cosine")
        warmup_steps = sched_config.get("warmup_steps", 50)

        if warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        else:
            warmup = None

        if name == "cosine":
            main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        elif name == "linear":
            main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)
        elif name == "cosine_with_restarts":
            num_cycles = sched_config.get("num_cycles", 1)
            T_0 = max(1, (total_steps - warmup_steps) // num_cycles)
            main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
        elif name == "constant":
            main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps)
        else:
            raise ValueError(f"Unsupported scheduler: {name}")

        if warmup and warmup_steps > 0:
            return SequentialLR(optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_steps])
        return main_scheduler

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        augmentation_fn=None,
    ) -> dict[str, Any]:
        """Run the training loop for a fixed step budget.

        Args:
            model: The model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (optional).
            augmentation_fn: Optional callable for text augmentation.

        Returns:
            Dictionary with training results and metrics.
        """
        training_config = self.config.get("training", {})
        reg_config = self.config.get("regularization", {})
        max_steps = training_config.get("max_steps", 500)
        eval_every = training_config.get("eval_every", 100)
        use_fp16 = training_config.get("fp16", True) and self.device.type == "cuda"
        max_grad_norm = reg_config.get("max_grad_norm", 1.0)
        grad_accum_steps = self.config.get("batch", {}).get("gradient_accumulation_steps", 1)

        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer, max_steps)
        scaler = GradScaler() if use_fp16 else None

        model.train()
        start_time = time.time()
        global_step = 0
        train_loss_accum = 0.0
        eval_results = []

        progress = Progress(
            TextColumn("[bold blue]Training"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[green]loss: {task.fields[loss]:.4f}"),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task("train", total=max_steps, loss=0.0)

            while global_step < max_steps:
                for batch in train_loader:
                    if global_step >= max_steps:
                        break

                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # Forward pass
                    with autocast(enabled=use_fp16):
                        if self.task in ("language_modeling", "text_classification"):
                            outputs = model(**batch)
                            loss = outputs.loss
                        elif self.task == "image_classification":
                            logits = model(batch["pixel_values"])
                            loss = nn.functional.cross_entropy(logits, batch["labels"])
                        else:
                            raise ValueError(f"Unsupported task: {self.task}")

                        loss = loss / grad_accum_steps

                    # Backward pass
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (global_step + 1) % grad_accum_steps == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        optimizer.zero_grad()

                    current_loss = loss.item() * grad_accum_steps
                    train_loss_accum += current_loss
                    global_step += 1

                    # Update progress
                    progress.update(task, advance=1, loss=current_loss)

                    # Log training metrics
                    self.metric_tracker.update(
                        global_step,
                        train_loss=current_loss,
                        learning_rate=scheduler.get_last_lr()[0],
                    )

                    if self.tracker:
                        self.tracker.log_metrics(global_step, {
                            "train/loss": current_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        })

                    # Evaluation
                    if val_loader and global_step % eval_every == 0:
                        eval_metrics = self.evaluate(model, val_loader)
                        eval_results.append({"step": global_step, **eval_metrics})

                        # Update metric tracker
                        self.metric_tracker.update(global_step, **{
                            f"eval_{k}": v for k, v in eval_metrics.items()
                        })

                        if self.tracker:
                            self.tracker.log_metrics(global_step, {
                                f"eval/{k}": v for k, v in eval_metrics.items()
                            })

                        model.train()

        # Final evaluation
        final_eval = {}
        if val_loader:
            final_eval = self.evaluate(model, val_loader)

        elapsed = time.time() - start_time
        avg_train_loss = train_loss_accum / max(global_step, 1)

        results = {
            "train_loss": avg_train_loss,
            "eval_metrics": final_eval,
            "eval_history": eval_results,
            "steps_completed": global_step,
            "elapsed_seconds": round(elapsed, 2),
            "metric_summary": self.metric_tracker.summary(),
        }

        # Add primary metric to top level for easy access
        if self.primary_metric.startswith("eval_"):
            metric_key = self.primary_metric.replace("eval_", "")
            results["primary_metric_value"] = final_eval.get(metric_key, avg_train_loss)
        else:
            results["primary_metric_value"] = avg_train_loss

        logger.info(
            f"Training complete: {global_step} steps, "
            f"train_loss={avg_train_loss:.4f}, "
            f"elapsed={elapsed:.1f}s"
        )

        return results

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on a dataset split.

        Returns:
            Dict of metric name → value.
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0
        max_eval = self.config.get("training", {}).get("max_eval_steps")

        for batch in dataloader:
            if max_eval and n_batches >= max_eval:
                break
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if self.task in ("language_modeling", "text_classification"):
                outputs = model(**batch)
                loss = outputs.loss.item()
                logits = outputs.logits
            elif self.task == "image_classification":
                logits = model(batch["pixel_values"])
                loss = nn.functional.cross_entropy(logits, batch["labels"]).item()
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            total_loss += loss
            n_batches += 1

            # Accuracy
            if self.task in ("text_classification", "image_classification"):
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch["labels"]).sum().item()
                total_samples += batch["labels"].size(0)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"loss": avg_loss}

        if self.task == "language_modeling":
            metrics["perplexity"] = compute_perplexity(avg_loss)
            metrics["bpb"] = compute_bpb(avg_loss)

        if total_samples > 0:
            metrics["accuracy"] = total_correct / total_samples

        return metrics

    def save_results(self, results: dict[str, Any], output_dir: str | Path) -> Path:
        """Save training results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results_path

    def save_model(self, model: nn.Module, output_dir: str | Path) -> Path:
        """Save the trained model or adapters.

        Args:
            model: The trained model to save.
            output_dir: Directory to save the model/adapters in.

        Returns:
            Path to the saved model directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model"

        # Check if LoRA was used
        is_lora = any(hasattr(m, "is_peft_model") or "PeftModel" in m.__class__.__name__ for m in model.modules())

        if is_lora:
            from src.training.lora import save_lora_weights
            save_lora_weights(model, model_path)
            logger.info(f"LoRA adapters saved to {model_path}")
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(model_path)
            # Also save tokenizer for NLP tasks
            model_name = self.config["model"]["name"]
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)
            logger.info(f"Transformers model and tokenizer saved to {model_path}")
        else:
            # Fallback for standard torch models
            torch.save(model.state_dict(), model_path / "model_weights.pt")
            logger.info(f"Model weights saved to {model_path / 'model_weights.pt'}")

        return model_path
