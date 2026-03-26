"""Metric computation utilities for training evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


def compute_bpb(loss: float, vocab_size: int = 50257) -> float:
    """Compute bits-per-byte from cross-entropy loss.

    BPB = loss * log2(e) / chars_per_token
    Approximation: assume ~4 chars per token for English text.
    """
    chars_per_token = 4.0  # Rough average for BPE tokenizers
    return loss * math.log2(math.e) / chars_per_token


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        # For language modeling, shift predictions
        if preds.dim() == 2 and labels.dim() == 2:
            preds = preds[:, :-1]
            labels = labels[:, 1:]
        correct = (preds == labels).float()
        # Ignore padding tokens (-100)
        mask = labels != -100
        if mask.any():
            return (correct * mask).sum().item() / mask.sum().item()
        return 0.0


@dataclass
class MetricTracker:
    """Track running metrics across training steps.

    Maintains running averages and tracks best values for each metric.
    """

    metrics: dict[str, list[float]] = field(default_factory=dict)
    best_values: dict[str, float] = field(default_factory=dict)
    best_steps: dict[str, int] = field(default_factory=dict)
    directions: dict[str, str] = field(default_factory=dict)  # minimize | maximize

    def set_direction(self, metric_name: str, direction: str) -> None:
        """Set optimization direction for a metric."""
        self.directions[metric_name] = direction

    def update(self, step: int, **kwargs: float) -> dict[str, bool]:
        """Update metrics and return dict of which metrics improved.

        Returns:
            Dict mapping metric names to whether they improved at this step.
        """
        improved = {}
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

            direction = self.directions.get(name, "minimize")
            is_better = self._is_better(value, self.best_values.get(name), direction)

            if is_better:
                self.best_values[name] = value
                self.best_steps[name] = step
                improved[name] = True
            else:
                improved[name] = False

        return improved

    def _is_better(
        self,
        current: float,
        best: float | None,
        direction: str,
    ) -> bool:
        """Check if current value is better than best."""
        if best is None:
            return True
        if direction == "minimize":
            return current < best
        return current > best

    def get_latest(self, metric_name: str) -> float | None:
        """Get the most recent value of a metric."""
        values = self.metrics.get(metric_name, [])
        return values[-1] if values else None

    def get_best(self, metric_name: str) -> tuple[float | None, int | None]:
        """Get the best value and step for a metric."""
        return self.best_values.get(metric_name), self.best_steps.get(metric_name)

    def get_running_average(self, metric_name: str, window: int = 10) -> float | None:
        """Get the running average of a metric over the last `window` values."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return None
        recent = values[-window:]
        return sum(recent) / len(recent)

    def summary(self) -> dict[str, Any]:
        """Get a summary of all tracked metrics."""
        result = {}
        for name in self.metrics:
            result[name] = {
                "latest": self.get_latest(name),
                "best": self.best_values.get(name),
                "best_step": self.best_steps.get(name),
                "avg_last_10": self.get_running_average(name, 10),
                "n_values": len(self.metrics[name]),
            }
        return result
