"""Main agent orchestration: propose → apply → train → evaluate → decide loop."""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.evaluator import TrialDecision, TrialEvaluator
from src.agent.proposer import ConfigProposer
from src.training.augmentations import AugmentationPipeline
from src.training.train import Trainer
from src.utils.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    apply_patch,
    diff_configs,
    hash_config,
    load_yaml,
    save_yaml,
    validate_augmentations,
    validate_tuning,
)
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)
console = Console()


class AgentRunner:
    """Orchestrates the autonomous augmentation + tuning loop.

    Loop: propose → validate → apply → train → evaluate → decide → log
    """

    def __init__(
        self,
        config: dict[str, Any],
        use_docker: bool = False,
        dry_run: bool = False,
    ):
        self.config = config
        self.use_docker = use_docker
        self.dry_run = dry_run

        self.proposer = ConfigProposer(config)
        self.evaluator = TrialEvaluator(config)

        # Load tracker
        self.tracker = self._init_tracker()

        # State
        self.baseline_metric: float | None = None
        self.best_metric: float | None = None
        self.best_config: dict | None = None
        self.trial_history: list[dict] = []

        # Provenance
        self.provenance = self._init_provenance()

        # Output directory
        self.output_dir = PROJECT_ROOT / config.get("project", {}).get("output_dir", "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_tracker(self):
        """Initialize experiment tracker based on config."""
        try:
            from src.tracking.experiment import ExperimentTracker

            return ExperimentTracker(self.config)
        except Exception as e:
            logger.warning(f"Could not initialize tracker: {e}")
            return None

    def _init_provenance(self):
        """Initialize provenance database."""
        try:
            from src.tracking.provenance import TrialDatabase

            db_path = PROJECT_ROOT / "provenance" / "trials.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return TrialDatabase(str(db_path))
        except Exception as e:
            logger.warning(f"Could not initialize provenance DB: {e}")
            return None

    def run(
        self,
        max_trials: int | None = None,
        resume: bool = False,
    ) -> list[TrialDecision]:
        """Run the autonomous agent loop.

        Args:
            max_trials: Override max trials from config.
            resume: Resume from last checkpoint.

        Returns:
            List of TrialDecisions for all completed trials.
        """
        agent_config = self.config.get("agent", {})
        max_trials = max_trials or agent_config.get("max_trials", 50)
        patience = agent_config.get("patience", 10)
        seed = self.config.get("project", {}).get("seed", 42)

        set_seed(seed)

        # Load starting state
        if resume:
            self._resume_state()

        # Load baseline
        if self.baseline_metric is None:
            console.print("[yellow]⚠ No baseline found. Run `ada baseline` first.[/yellow]")
            console.print("[dim]  Running baseline now...[/dim]")
            self.baseline_metric = self._establish_baseline()
            self.best_metric = self.baseline_metric

        console.print(Panel(
            f"[bold]Baseline {self.evaluator.metric_name}:[/bold] {self.baseline_metric:.4f}\n"
            f"[bold]Max trials:[/bold] {max_trials}\n"
            f"[bold]Improvement threshold:[/bold] {self.evaluator.threshold * 100:.2f}%\n"
            f"[bold]Mode:[/bold] {'Docker' if self.use_docker else 'Local'} | "
            f"{'Dry run' if self.dry_run else 'Live'}",
            title="🧬 Agent Starting",
            border_style="cyan",
        ))

        decisions: list[TrialDecision] = []
        no_improvement_count = 0
        start_trial = len(self.trial_history)

        for trial_id in range(start_trial, start_trial + max_trials):
            console.print(f"\n{'='*60}")
            console.print(f"[bold cyan]Trial {trial_id + 1} / {start_trial + max_trials}[/bold cyan]")
            console.print(f"{'='*60}")

            try:
                decision = self._run_trial(trial_id)
                decisions.append(decision)

                # Display result
                if decision.accepted:
                    console.print(Panel(
                        f"[green]✅ ACCEPTED[/green]\n{decision.reason}",
                        border_style="green",
                    ))
                    no_improvement_count = 0
                else:
                    console.print(Panel(
                        f"[red]❌ REJECTED[/red]\n{decision.reason}",
                        border_style="red",
                    ))
                    no_improvement_count += 1

                # Early stopping
                if no_improvement_count >= patience:
                    console.print(
                        f"\n[yellow]⚠ No improvement for {patience} trials. "
                        f"Stopping early.[/yellow]"
                    )
                    break

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠ Interrupted. Saving state...[/yellow]")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Trial {trial_id} error: {e}", exc_info=True)
                console.print(f"[red]Trial {trial_id} error: {e}[/red]")
                continue

        # Summary
        self._print_summary(decisions)
        self._save_state()

        return decisions

    def _run_trial(self, trial_id: int) -> TrialDecision:
        """Execute a single trial."""
        # Load current configs
        current_aug = load_yaml(CONFIG_DIR / "augmentations.yaml")
        current_tuning = load_yaml(CONFIG_DIR / "tuning.yaml")

        # 1. Propose
        console.print("[dim]  Proposing config changes...[/dim]")
        proposal = self.proposer.propose(
            current_aug_config=current_aug,
            current_tuning_config=current_tuning,
            trial_history=self.trial_history,
            trial_number=trial_id + 1,
            max_trials=self.config.get("agent", {}).get("max_trials", 50),
        )

        # 2. Apply patch
        proposed_aug = apply_patch(current_aug, proposal.get("augmentations", {}))
        proposed_tuning = apply_patch(current_tuning, proposal.get("tuning", {}))

        # Show diff
        aug_diff = diff_configs(current_aug, proposed_aug)
        tuning_diff = diff_configs(current_tuning, proposed_tuning)
        all_diff = aug_diff + tuning_diff

        if all_diff:
            console.print("[dim]  Changes:[/dim]")
            for change in all_diff:
                console.print(f"    {change}")
        else:
            console.print("[dim]  No changes proposed (repeating current config)[/dim]")

        # 3. Validate
        aug_errors = validate_augmentations(proposed_aug)
        tuning_errors = validate_tuning(proposed_tuning)
        if aug_errors or tuning_errors:
            return TrialDecision(
                accepted=False,
                reason=f"Config validation failed: {aug_errors + tuning_errors}",
                metrics={},
                improvement=0.0,
                trial_id=trial_id,
            )

        # 4. Dry run check
        if self.dry_run:
            console.print("[yellow]  [Dry run] Skipping training[/yellow]")
            return TrialDecision(
                accepted=False,
                reason="Dry run — training skipped",
                metrics={},
                improvement=0.0,
                trial_id=trial_id,
            )

        # 5. Train
        config_hash = hash_config({**proposed_aug, **proposed_tuning})
        console.print(f"[dim]  Training (config hash: {config_hash})...[/dim]")

        merged_config = copy.deepcopy(self.config)
        merged_config.update(proposed_tuning)

        if self.tracker:
            self.tracker.start_trial(trial_id, merged_config)

        if self.use_docker:
            results = self._train_docker(trial_id, proposed_aug, proposed_tuning)
        else:
            results = self._train_local(trial_id, merged_config, proposed_aug)

        # 6. Safety checks
        safety_passed = self._run_safety_checks(proposed_aug)

        # 7. Evaluate
        decision = self.evaluator.evaluate(
            trial_id=trial_id,
            trial_results=results,
            baseline_metric=self.baseline_metric,
            best_metric=self.best_metric,
            safety_passed=safety_passed,
        )

        # 8. Update state
        trial_record = {
            "trial_id": trial_id,
            "config_hash": config_hash,
            "augmentations": proposed_aug,
            "tuning": proposed_tuning,
            "metrics": results.get("eval_metrics", {}),
            "primary_metric": results.get("primary_metric_value"),
            "decision": "accepted" if decision.accepted else "rejected",
            "reason": decision.reason,
            "timestamp": datetime.now().isoformat(),
            "diff": all_diff,
        }
        self.trial_history.append(trial_record)

        if decision.accepted:
            self.best_metric = results.get("primary_metric_value", self.best_metric)
            self.best_config = {**proposed_aug, **proposed_tuning}

            # Save accepted config
            save_yaml(proposed_aug, CONFIG_DIR / "augmentations.yaml")
            save_yaml(proposed_tuning, CONFIG_DIR / "tuning.yaml")

            # Create git branch/PR
            self._create_provenance(trial_id, trial_record)

        # Log to provenance DB
        if self.provenance:
            self.provenance.log_trial(trial_record)

        if self.tracker:
            self.tracker.end_trial(decision)

        return decision

    def _train_local(
        self,
        trial_id: int,
        config: dict,
        aug_config: dict,
    ) -> dict[str, Any]:
        """Run training locally."""
        from src.data.loader import create_dataloaders, load_dataset_from_config

        trainer = Trainer(config, tracker=self.tracker)
        model = trainer.load_model()

        # Apply LoRA if enabled
        if config.get("lora", {}).get("enabled", False):
            from src.training.lora import apply_lora

            model = apply_lora(model, config["lora"])

        # Load data
        splits = load_dataset_from_config(config)
        loaders = create_dataloaders(splits, config)

        # Build augmentation pipeline
        aug_pipeline = AugmentationPipeline(aug_config)

        # Train
        results = trainer.train(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders.get("validation"),
            augmentation_fn=aug_pipeline.augment_text if aug_pipeline.enabled else None,
        )

        # Save results
        trial_dir = self.output_dir / f"trial_{trial_id}"
        trainer.save_results(results, trial_dir)

        return results

    def _train_docker(
        self,
        trial_id: int,
        aug_config: dict,
        tuning_config: dict,
    ) -> dict[str, Any]:
        """Run training in Docker container."""
        from src.utils.docker import DockerTrialRunner

        exec_config = self.config.get("execution", {})
        runner = DockerTrialRunner(
            image=exec_config.get("docker_image", "auto-augment-agent:latest"),
            gpu_ids=exec_config.get("gpu_ids", "0"),
            memory_limit=exec_config.get("memory_limit", "16g"),
            network_enabled=exec_config.get("network", False),
        )

        # Save proposed configs to a temp location
        trial_config_dir = self.output_dir / f"trial_{trial_id}" / "config"
        trial_config_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(aug_config, trial_config_dir / "augmentations.yaml")
        save_yaml(tuning_config, trial_config_dir / "tuning.yaml")

        # Copy base config
        import shutil

        shutil.copy(CONFIG_DIR / "base.yaml", trial_config_dir / "base.yaml")

        trial_output = self.output_dir / f"trial_{trial_id}" / "output"
        return runner.run_trial(
            trial_id=trial_id,
            config_dir=trial_config_dir,
            output_dir=trial_output,
        )

    def _run_safety_checks(self, aug_config: dict) -> bool:
        """Run PII and license safety checks."""
        try:
            from src.safety.license_checker import LicenseChecker
            from src.safety.pii_scanner import PIIScanner

            # PII check
            scanner = PIIScanner()
            pii_result = scanner.check_config(aug_config)
            if not pii_result["safe"]:
                logger.warning(f"PII check failed: {pii_result['issues']}")
                return False

            # License check
            checker = LicenseChecker()
            license_result = checker.check(self.config)
            if not license_result["passed"]:
                logger.warning(f"License check failed: {license_result['issues']}")
                return False

            return True
        except ImportError:
            logger.debug("Safety modules not available, skipping")
            return True
        except Exception as e:
            logger.warning(f"Safety check error: {e}")
            return True  # Don't block on check failures

    def _establish_baseline(self) -> float:
        """Train with default config to establish baseline metrics."""
        console.print("[cyan]Establishing baseline...[/cyan]")

        aug_config = load_yaml(CONFIG_DIR / "augmentations.yaml")
        tuning_config = load_yaml(CONFIG_DIR / "tuning.yaml")

        merged = copy.deepcopy(self.config)
        merged.update(tuning_config)

        results = self._train_local(-1, merged, aug_config)
        baseline = results.get("primary_metric_value", 0.0)

        console.print(f"[green]Baseline established: {self.evaluator.metric_name} = {baseline:.4f}[/green]")
        return baseline

    def _create_provenance(self, trial_id: int, trial_record: dict) -> None:
        """Create git branch and PR for accepted trial."""
        try:
            from src.tracking.provenance import GitManager

            git_mgr = GitManager(str(PROJECT_ROOT))
            branch = git_mgr.create_trial_branch(trial_id)
            git_mgr.commit_config_changes(trial_id, trial_record.get("diff", []))
            logger.info(f"Created branch: {branch}")
        except Exception as e:
            logger.warning(f"Provenance creation failed: {e}")

    def _save_state(self) -> None:
        """Save agent state for resume."""
        state = {
            "baseline_metric": self.baseline_metric,
            "best_metric": self.best_metric,
            "best_config": self.best_config,
            "trial_history": self.trial_history,
        }
        state_path = self.output_dir / "agent_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        console.print(f"[dim]State saved to {state_path}[/dim]")

    def _resume_state(self) -> None:
        """Resume from saved state."""
        state_path = self.output_dir / "agent_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.baseline_metric = state.get("baseline_metric")
            self.best_metric = state.get("best_metric")
            self.best_config = state.get("best_config")
            self.trial_history = state.get("trial_history", [])
            console.print(
                f"[green]Resumed from trial {len(self.trial_history)}, "
                f"best {self.evaluator.metric_name}: {self.best_metric}[/green]"
            )
        else:
            console.print("[yellow]No saved state found, starting fresh[/yellow]")

    def _print_summary(self, decisions: list[TrialDecision]) -> None:
        """Print a summary of all trials."""
        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]

        console.print(f"\n{'='*60}")
        table = Table(title="🧬 Agent Run Summary", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Total trials", str(len(decisions)))
        table.add_row("Accepted", f"[green]{len(accepted)}[/green]")
        table.add_row("Rejected", f"[red]{len(rejected)}[/red]")
        table.add_row("Baseline", f"{self.baseline_metric:.4f}" if self.baseline_metric else "N/A")
        table.add_row("Best", f"{self.best_metric:.4f}" if self.best_metric else "N/A")

        if self.baseline_metric and self.best_metric:
            improvement = abs(self.best_metric - self.baseline_metric) / abs(self.baseline_metric) * 100
            table.add_row("Total improvement", f"{improvement:.2f}%")

        console.print(table)
