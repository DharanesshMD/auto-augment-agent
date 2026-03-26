"""LLM-based configuration proposal generator.

Uses the LLM (via litellm) to propose augmentation and hyperparameter
changes based on trial history. Falls back to random mutation if LLM fails.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import SCHEMA_DIR, validate_augmentations, validate_tuning

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class ConfigProposer:
    """Proposes config changes using an LLM with trial history context.

    The proposer:
    1. Constructs a prompt with current config, trial history, and schema constraints
    2. Calls the LLM to generate a YAML patch
    3. Validates the patch against JSON schemas
    4. Falls back to random mutation if LLM fails
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.provider = self.llm_config.get("provider", "ollama")
        self.model = self.llm_config.get("model", "llama3")
        self.temperature = self.llm_config.get("temperature", 0.7)
        self.max_tokens = self.llm_config.get("max_tokens", 2048)

        # Load prompt templates
        self.system_prompt = self._load_prompt("system.md")
        self.propose_template = self._load_prompt("propose.md")

        # Load schemas for context
        self.aug_schema = self._load_schema("augmentations_schema.json")
        self.tuning_schema = self._load_schema("tuning_schema.json")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template."""
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text()
        logger.warning(f"Prompt template not found: {path}")
        return ""

    def _load_schema(self, filename: str) -> dict:
        """Load a JSON schema."""
        path = SCHEMA_DIR / filename
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def propose(
        self,
        current_aug_config: dict[str, Any],
        current_tuning_config: dict[str, Any],
        trial_history: list[dict[str, Any]],
        trial_number: int,
        max_trials: int,
    ) -> dict[str, Any]:
        """Propose new augmentation and tuning configurations.

        Args:
            current_aug_config: Current augmentation YAML config.
            current_tuning_config: Current tuning YAML config.
            trial_history: List of past trial results.
            trial_number: Current trial number (for exploration vs. exploitation).
            max_trials: Total trials planned.

        Returns:
            Dict with 'augmentations' and 'tuning' patches.
        """
        # Determine exploration vs exploitation phase
        exploration_trials = self.config.get("agent", {}).get("exploration_trials", 10)
        is_exploring = trial_number <= exploration_trials

        try:
            proposal = self._llm_propose(
                current_aug_config,
                current_tuning_config,
                trial_history,
                trial_number,
                max_trials,
                is_exploring,
            )

            # Validate proposal
            if "augmentations" in proposal:
                errors = validate_augmentations(proposal["augmentations"])
                if errors:
                    logger.warning(f"LLM proposal validation errors (aug): {errors}")
                    proposal["augmentations"] = current_aug_config

            if "tuning" in proposal:
                errors = validate_tuning(proposal["tuning"])
                if errors:
                    logger.warning(f"LLM proposal validation errors (tuning): {errors}")
                    proposal["tuning"] = current_tuning_config

            return proposal

        except Exception as e:
            logger.warning(f"LLM proposal failed ({e}), falling back to random mutation")
            return self._random_mutate(
                current_aug_config,
                current_tuning_config,
                is_exploring,
            )

    def _llm_propose(
        self,
        current_aug: dict,
        current_tuning: dict,
        history: list[dict],
        trial_num: int,
        max_trials: int,
        is_exploring: bool,
    ) -> dict[str, Any]:
        """Generate a proposal using the LLM."""
        import litellm

        # Build context
        recent_history = history[-5:] if len(history) > 5 else history
        history_str = yaml.dump(recent_history, default_flow_style=False) if recent_history else "No trials yet."

        best_trial = None
        if history:
            metric = self.config.get("agent", {}).get("metric", "eval_loss")
            direction = self.config.get("agent", {}).get("metric_direction", "minimize")
            best_trial = sorted(
                [h for h in history if h.get("metrics")],
                key=lambda h: h["metrics"].get(metric, float("inf") if direction == "minimize" else float("-inf")),
                reverse=(direction == "maximize"),
            )
            best_trial = best_trial[0] if best_trial else None

        prompt = self.propose_template.format(
            current_augmentations=yaml.dump(current_aug, default_flow_style=False),
            current_tuning=yaml.dump(current_tuning, default_flow_style=False),
            trial_history=history_str,
            best_trial=yaml.dump(best_trial, default_flow_style=False) if best_trial else "None yet",
            trial_number=trial_num,
            max_trials=max_trials,
            phase="EXPLORATION (try diverse changes)" if is_exploring else "EXPLOITATION (refine best config)",
            augmentation_schema=json.dumps(self.aug_schema, indent=2)[:2000],
            tuning_schema=json.dumps(self.tuning_schema, indent=2)[:2000],
        )

        # Determine model string for litellm
        if self.provider == "ollama":
            model_str = f"ollama/{self.model}"
        elif self.provider == "openai":
            model_str = self.model
        elif self.provider == "anthropic":
            model_str = self.model
        else:
            model_str = self.model

        response = litellm.completion(
            model=model_str,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        return self._parse_proposal(content)

    def _parse_proposal(self, llm_output: str) -> dict[str, Any]:
        """Parse LLM output into augmentation and tuning config patches."""
        # Try to extract YAML blocks from the response
        proposal = {"augmentations": {}, "tuning": {}}

        # Look for YAML between ```yaml ... ``` blocks
        import re

        yaml_blocks = re.findall(r"```(?:yaml)?\s*\n(.*?)```", llm_output, re.DOTALL)

        for block in yaml_blocks:
            try:
                parsed = yaml.safe_load(block)
                if not isinstance(parsed, dict):
                    continue

                # Detect if it's augmentation or tuning config
                if "pipeline" in parsed:
                    proposal["augmentations"] = parsed
                elif "optimizer" in parsed or "scheduler" in parsed or "batch" in parsed:
                    proposal["tuning"] = parsed
                elif "augmentations" in parsed:
                    proposal["augmentations"] = parsed["augmentations"]
                elif "tuning" in parsed:
                    proposal["tuning"] = parsed["tuning"]
                else:
                    # Try to merge into tuning as default
                    proposal["tuning"].update(parsed)
            except yaml.YAMLError:
                continue

        # If no YAML blocks found, try parsing the entire response
        if not proposal["augmentations"] and not proposal["tuning"]:
            try:
                parsed = yaml.safe_load(llm_output)
                if isinstance(parsed, dict):
                    proposal["augmentations"] = parsed.get("augmentations", {})
                    proposal["tuning"] = parsed.get("tuning", {})
            except yaml.YAMLError:
                logger.warning("Could not parse LLM output as YAML")

        return proposal

    def _random_mutate(
        self,
        current_aug: dict,
        current_tuning: dict,
        is_exploring: bool,
    ) -> dict[str, Any]:
        """Generate a random mutation as fallback.

        More aggressive mutations during exploration, conservative during exploitation.
        """
        import copy

        aug_patch = copy.deepcopy(current_aug)
        tuning_patch = copy.deepcopy(current_tuning)

        # Mutate learning rate
        lr = tuning_patch.get("optimizer", {}).get("learning_rate", 5e-5)
        if is_exploring:
            factor = random.choice([0.1, 0.3, 0.5, 2.0, 3.0, 10.0])
        else:
            factor = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.5])
        new_lr = max(1e-7, min(1e-2, lr * factor))
        tuning_patch.setdefault("optimizer", {})["learning_rate"] = new_lr

        # Randomly toggle an augmentation
        text_augs = aug_patch.get("pipeline", {}).get("text_augmentations", [])
        if text_augs:
            idx = random.randint(0, len(text_augs) - 1)
            text_augs[idx]["enabled"] = not text_augs[idx].get("enabled", False)
            # Randomize probability if enabling
            if text_augs[idx]["enabled"]:
                text_augs[idx]["probability"] = round(random.uniform(0.05, 0.3), 2)
            aug_patch.setdefault("pipeline", {})["enabled"] = any(
                a.get("enabled", False) for a in text_augs
            )

        # Maybe mutate batch size
        if random.random() < 0.3:
            batch_sizes = [2, 4, 8, 16, 32]
            tuning_patch.setdefault("batch", {})["train_batch_size"] = random.choice(batch_sizes)

        # Maybe mutate dropout
        if random.random() < 0.3:
            new_dropout = round(random.uniform(0.0, 0.4), 2)
            tuning_patch.setdefault("regularization", {})["dropout"] = new_dropout

        # Maybe toggle LoRA
        if self.config.get("modules", {}).get("lora", False) and random.random() < 0.2:
            lora = tuning_patch.setdefault("lora", {})
            lora["enabled"] = not lora.get("enabled", False)
            if lora["enabled"]:
                lora["rank"] = random.choice([4, 8, 16, 32])
                lora["alpha"] = lora["rank"] * 2

        return {"augmentations": aug_patch, "tuning": tuning_patch}
