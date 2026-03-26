"""Tests for the trial evaluator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.evaluator import TrialEvaluator


@pytest.fixture
def evaluator():
    return TrialEvaluator({
        "agent": {
            "metric": "eval_loss",
            "metric_direction": "minimize",
            "improvement_threshold": 0.005,
        }
    })


@pytest.fixture
def maximize_evaluator():
    return TrialEvaluator({
        "agent": {
            "metric": "eval_accuracy",
            "metric_direction": "maximize",
            "improvement_threshold": 0.005,
        }
    })


class TestEvaluator:
    def test_first_trial_accepted(self, evaluator):
        results = {
            "eval_metrics": {"loss": 3.5},
            "primary_metric_value": 3.5,
        }
        decision = evaluator.evaluate(0, results, None, None)
        assert decision.accepted

    def test_improved_trial_accepted(self, evaluator):
        results = {
            "eval_metrics": {"loss": 3.0},
            "primary_metric_value": 3.0,
        }
        decision = evaluator.evaluate(1, results, baseline_metric=3.5, best_metric=3.5)
        assert decision.accepted
        assert decision.improvement > 0

    def test_no_improvement_rejected(self, evaluator):
        results = {
            "eval_metrics": {"loss": 3.5},
            "primary_metric_value": 3.5,
        }
        decision = evaluator.evaluate(1, results, baseline_metric=3.5, best_metric=3.5)
        assert not decision.accepted

    def test_worse_trial_rejected(self, evaluator):
        results = {
            "eval_metrics": {"loss": 4.0},
            "primary_metric_value": 4.0,
        }
        decision = evaluator.evaluate(1, results, baseline_metric=3.5, best_metric=3.5)
        assert not decision.accepted

    def test_failed_trial_rejected(self, evaluator):
        results = {"status": "error", "error": "OOM"}
        decision = evaluator.evaluate(1, results, baseline_metric=3.5, best_metric=3.5)
        assert not decision.accepted
        assert "failed" in decision.reason.lower() or "error" in decision.reason.lower()

    def test_safety_failure_rejects(self, evaluator):
        results = {
            "eval_metrics": {"loss": 2.0},
            "primary_metric_value": 2.0,
        }
        decision = evaluator.evaluate(
            1, results, baseline_metric=3.5, best_metric=3.5, safety_passed=False
        )
        assert not decision.accepted
        assert "safety" in decision.reason.lower()

    def test_maximize_direction(self, maximize_evaluator):
        results = {
            "eval_metrics": {"accuracy": 0.95},
            "primary_metric_value": 0.95,
        }
        decision = maximize_evaluator.evaluate(1, results, baseline_metric=0.90, best_metric=0.90)
        assert decision.accepted

    def test_maximize_no_improvement(self, maximize_evaluator):
        results = {
            "eval_metrics": {"accuracy": 0.90},
            "primary_metric_value": 0.90,
        }
        decision = maximize_evaluator.evaluate(1, results, baseline_metric=0.90, best_metric=0.90)
        assert not decision.accepted

    def test_below_threshold_rejected(self, evaluator):
        # 0.1% improvement, threshold is 0.5%
        results = {
            "eval_metrics": {"loss": 3.4965},
            "primary_metric_value": 3.4965,
        }
        decision = evaluator.evaluate(1, results, baseline_metric=3.5, best_metric=3.5)
        assert not decision.accepted
