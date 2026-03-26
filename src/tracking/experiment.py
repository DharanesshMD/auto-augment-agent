"""Experiment tracking — unified interface for W&B, MLflow, and local JSON."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface.

    Supports: Weights & Biases, MLflow, and local JSON logging.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        tracking_config = config.get("tracking", {})
        self.backend = tracking_config.get("backend", "none")
        self.project = tracking_config.get("project", "auto-augment-agent")
        self.entity = tracking_config.get("entity")

        self._run = None
        self._backend_impl = self._init_backend()

    def _init_backend(self):
        """Initialize the tracking backend."""
        if self.backend == "wandb":
            return _WandbBackend(self.project, self.entity)
        elif self.backend == "mlflow":
            return _MLflowBackend(self.project, self.config.get("tracking", {}).get("entity"))
        else:
            return _LocalBackend(self.project)

    def start_trial(self, trial_id: int, config: dict[str, Any]) -> None:
        """Start tracking a new trial."""
        self._backend_impl.start_run(
            name=f"trial-{trial_id}",
            config=config,
            tags={"trial_id": str(trial_id)},
        )

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics at a training step."""
        self._backend_impl.log_metrics(step, metrics)

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Log an artifact (model checkpoint, config, etc.)."""
        self._backend_impl.log_artifact(path, artifact_type)

    def end_trial(self, decision) -> None:
        """Finalize the trial run."""
        self._backend_impl.end_run(
            accepted=decision.accepted,
            reason=decision.reason,
        )


class _WandbBackend:
    """Weights & Biases backend."""

    def __init__(self, project: str, entity: str | None):
        self.project = project
        self.entity = entity
        self._run = None

    def start_run(self, name: str, config: dict, tags: dict) -> None:
        import wandb

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=list(tags.values()),
            reinit=True,
        )

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        if self._run:
            import wandb

            wandb.log(metrics, step=step)

    def log_artifact(self, path: str, artifact_type: str) -> None:
        if self._run:
            import wandb

            artifact = wandb.Artifact(
                name=Path(path).stem,
                type=artifact_type,
            )
            if Path(path).is_dir():
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            self._run.log_artifact(artifact)

    def end_run(self, accepted: bool, reason: str) -> None:
        if self._run:
            import wandb

            wandb.log({"accepted": accepted, "reason": reason})
            wandb.finish()
            self._run = None


class _MLflowBackend:
    """MLflow backend."""

    def __init__(self, project: str, tracking_uri: str | None):
        self.project = project
        self._run = None

        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(project)

    def start_run(self, name: str, config: dict, tags: dict) -> None:
        import mlflow

        self._run = mlflow.start_run(run_name=name, tags=tags)
        mlflow.log_params(_flatten_dict(config, max_depth=2))

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        import mlflow

        # MLflow doesn't support slashes in metric names
        clean_metrics = {k.replace("/", "."): v for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, path: str, artifact_type: str) -> None:
        import mlflow

        if Path(path).is_dir():
            mlflow.log_artifacts(path)
        else:
            mlflow.log_artifact(path)

    def end_run(self, accepted: bool, reason: str) -> None:
        import mlflow

        mlflow.log_metrics({"accepted": int(accepted)})
        mlflow.set_tag("decision", "accepted" if accepted else "rejected")
        mlflow.set_tag("reason", reason[:250])
        mlflow.end_run()
        self._run = None


class _LocalBackend:
    """Local JSON file logging backend (no external dependencies)."""

    def __init__(self, project: str):
        self.project = project
        self.log_dir = Path("logs") / project
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: dict[str, Any] = {}
        self._run_file: Path | None = None

    def start_run(self, name: str, config: dict, tags: dict) -> None:
        self._current_run = {
            "name": name,
            "config": config,
            "tags": tags,
            "metrics": [],
            "started_at": datetime.now().isoformat(),
        }
        self._run_file = self.log_dir / f"{name}.json"

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        self._current_run.setdefault("metrics", []).append({
            "step": step,
            **metrics,
        })

    def log_artifact(self, path: str, artifact_type: str) -> None:
        self._current_run.setdefault("artifacts", []).append({
            "path": path,
            "type": artifact_type,
        })

    def end_run(self, accepted: bool, reason: str) -> None:
        self._current_run["accepted"] = accepted
        self._current_run["reason"] = reason
        self._current_run["ended_at"] = datetime.now().isoformat()

        if self._run_file:
            with open(self._run_file, "w") as f:
                json.dump(self._current_run, f, indent=2, default=str)

        self._current_run = {}
        self._run_file = None


def _flatten_dict(d: dict, prefix: str = "", max_depth: int = 3) -> dict:
    """Flatten a nested dict for MLflow param logging."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and max_depth > 0:
            items.update(_flatten_dict(v, f"{new_key}.", max_depth - 1))
        else:
            items[new_key] = str(v)[:250]  # MLflow param value limit
    return items
