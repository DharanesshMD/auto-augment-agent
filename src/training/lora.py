"""LoRA (Low-Rank Adaptation) integration using HuggingFace PEFT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    config: dict[str, Any],
) -> nn.Module:
    """Wrap a model with LoRA adapters using PEFT.

    Args:
        model: The base model to add LoRA adapters to.
        config: LoRA configuration from tuning.yaml containing:
            - rank: LoRA rank (e.g., 8)
            - alpha: LoRA alpha scaling (e.g., 16)
            - target_modules: List of module names to apply LoRA to
            - dropout: LoRA dropout rate
            - bias: Bias training strategy ("none", "all", "lora_only")

    Returns:
        Model with LoRA adapters applied.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError(
            "peft library required for LoRA. Install with: pip install peft"
        )

    # Determine task type from model architecture
    task_type = _infer_task_type(model)

    lora_config = LoraConfig(
        task_type=task_type,
        r=config.get("rank", 8),
        lora_alpha=config.get("alpha", 16),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("dropout", 0.05),
        bias=config.get("bias", "none"),
    )

    peft_model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable, total = _count_parameters(peft_model)
    logger.info(
        f"LoRA applied: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return peft_model


def _infer_task_type(model: nn.Module):
    """Infer PEFT TaskType from the model class."""
    from peft import TaskType

    model_class = model.__class__.__name__.lower()

    if "causal" in model_class or "gpt" in model_class:
        return TaskType.CAUSAL_LM
    elif "seq2seq" in model_class or "t5" in model_class or "bart" in model_class:
        return TaskType.SEQ_2_SEQ_LM
    elif "sequence" in model_class or "classification" in model_class:
        return TaskType.SEQ_CLS
    elif "token" in model_class:
        return TaskType.TOKEN_CLS
    else:
        # Default to causal LM for generic models
        return TaskType.CAUSAL_LM


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_weights(model: nn.Module, save_path: str | Path) -> None:
    """Save only the LoRA adapter weights (much smaller than full model).

    Args:
        model: PEFT model with LoRA adapters.
        save_path: Directory to save adapter weights.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained(str(save_path))
        logger.info(f"LoRA weights saved to: {save_path}")
    except AttributeError:
        # Fallback: save state dict of trainable params only
        trainable_state = {
            k: v for k, v in model.state_dict().items() if "lora" in k.lower()
        }
        torch.save(trainable_state, save_path / "lora_weights.pt")
        logger.info(f"LoRA weights saved (fallback) to: {save_path / 'lora_weights.pt'}")


def load_lora_weights(
    model: nn.Module,
    load_path: str | Path,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Load LoRA adapter weights onto a base model.

    Args:
        model: Base model (without LoRA adapters).
        load_path: Directory containing adapter weights.
        config: Optional LoRA config if model doesn't have adapters yet.

    Returns:
        Model with loaded LoRA weights.
    """
    load_path = Path(load_path)

    try:
        from peft import PeftModel

        # Load PEFT model from saved adapters
        peft_model = PeftModel.from_pretrained(model, str(load_path))
        logger.info(f"LoRA weights loaded from: {load_path}")
        return peft_model
    except Exception as e:
        logger.warning(f"PEFT loading failed ({e}), trying fallback...")

        # Fallback: apply LoRA config then load state dict
        if config:
            model = apply_lora(model, config)

        weights_file = load_path / "lora_weights.pt"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"LoRA weights loaded (fallback) from: {weights_file}")

        return model
