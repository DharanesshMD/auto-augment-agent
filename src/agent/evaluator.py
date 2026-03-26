"""Trial evaluator with improvement gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrialDecision:
    """Result of evaluating a trial."""

    accepted: bool
    reason: str
    metrics: dict[str, float]
    improvement: float  # Relative improvement over baseline/best
    trial_id: int


class TrialEvaluator:
    """Evaluates trial results and decides whether to accept changes.

    Acceptance criteria:
    1. Trial must complete successfully (no training errors)
    2. Primary metric must improve by >= threshold (relative)
    3. Safety checks must pass (PII, license)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        agent_config = config.get("agent", {})
        self.metric_name = agent_config.get("metric", "eval_loss")
        self.direction = agent_config.get("metric_direction", "minimize")
        self.threshold = agent_config.get("improvement_threshold", 0.005)

    def evaluate(
        self,
        trial_id: int,
        trial_results: dict[str, Any],
        baseline_metric: float | None,
        best_metric: float | None,
        safety_passed: bool = True,
    ) -> TrialDecision:
        """Evaluate a completed trial and decide whether to accept.

        Args:
            trial_id: Trial identifier.
            trial_results: Results from the training run.
            baseline_metric: The baseline metric value (initial, no augmentations).
            best_metric: The best metric value seen so far.
            safety_passed: Whether safety checks (PII, license) passed.

        Returns:
            TrialDecision with accept/reject and reasoning.
        """
        # Check for training failure
        if trial_results.get("status") == "error" or trial_results.get("status") == "failed":
            return TrialDecision(
                accepted=False,
                reason=f"Trial failed: {trial_results.get('error', 'unknown error')}",
                metrics={},
                improvement=0.0,
                trial_id=trial_id,
            )

        # Extract the primary metric
        eval_metrics = trial_results.get("eval_metrics", {})
        primary_value = trial_results.get("primary_metric_value")

        if primary_value is None:
            # Try to find it in eval_metrics
            metric_key = self.metric_name.replace("eval_", "")
            primary_value = eval_metrics.get(metric_key)

        if primary_value is None:
            return TrialDecision(
                accepted=False,
                reason=f"Primary metric '{self.metric_name}' not found in results",
                metrics=eval_metrics,
                improvement=0.0,
                trial_id=trial_id,
            )

        # Safety check
        if not safety_passed:
            return TrialDecision(
                accepted=False,
                reason="Safety checks failed (PII detected or license incompatible)",
                metrics=eval_metrics,
                improvement=0.0,
                trial_id=trial_id,
            )

        # Calculate improvement against best (or baseline if no best yet)
        compare_to = best_metric if best_metric is not None else baseline_metric

        if compare_to is None:
            # First trial — auto-accept as baseline
            return TrialDecision(
                accepted=True,
                reason="First trial — accepted as initial baseline",
                metrics=eval_metrics,
                improvement=0.0,
                trial_id=trial_id,
            )

        improvement = self._calculate_improvement(primary_value, compare_to)

        # Check against threshold
        if improvement >= self.threshold:
            return TrialDecision(
                accepted=True,
                reason=(
                    f"Improved {self.metric_name} by {improvement:.4f} "
                    f"({improvement * 100:.2f}%) over {'best' if best_metric is not None else 'baseline'} "
                    f"({compare_to:.4f} → {primary_value:.4f})"
                ),
                metrics=eval_metrics,
                improvement=improvement,
                trial_id=trial_id,
            )
        else:
            return TrialDecision(
                accepted=False,
                reason=(
                    f"Insufficient improvement: {improvement:.4f} "
                    f"({improvement * 100:.2f}%) < threshold {self.threshold * 100:.2f}% "
                    f"(current: {primary_value:.4f}, best: {compare_to:.4f})"
                ),
                metrics=eval_metrics,
                improvement=improvement,
                trial_id=trial_id,
            )

    def _calculate_improvement(self, current: float, reference: float) -> float:
        """Calculate relative improvement.

        For 'minimize': improvement = (reference - current) / |reference|
        For 'maximize': improvement = (current - reference) / |reference|
        """
        if reference == 0:
            return 0.0

        if self.direction == "minimize":
            return (reference - current) / abs(reference)
        else:
            return (current - reference) / abs(reference)
