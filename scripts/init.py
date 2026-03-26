"""Interactive setup wizard for Auto-Augment Agent.

Run with: ada init
Or:       ada init --preset nlp-quick
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

console = Console()

# ─── Preset Definitions ─────────────────────────────────────────────────────

PRESETS = {
    "nlp-quick": {
        "description": "GPT-2 + WikiText-2, local Ollama, JSON logs, local execution",
        "model": {"name": "gpt2", "task": "language_modeling", "pretrained": True},
        "dataset": {
            "name": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "max_length": 512,
        },
        "llm": {"provider": "ollama", "model": "llama3", "temperature": 0.7},
        "tracking": {"backend": "none"},
        "execution": {"mode": "local"},
        "modules": {"web_fetcher": False, "lora": False},
        "agent": {"metric": "eval_loss", "metric_direction": "minimize"},
    },
    "cv-quick": {
        "description": "ResNet-18 + CIFAR-10, local Ollama, JSON logs, local execution",
        "model": {"name": "resnet18", "task": "image_classification", "pretrained": True},
        "dataset": {"name": "cifar10", "image_size": 32},
        "llm": {"provider": "ollama", "model": "llama3", "temperature": 0.7},
        "tracking": {"backend": "none"},
        "execution": {"mode": "local"},
        "modules": {"web_fetcher": False, "lora": False},
        "agent": {"metric": "eval_accuracy", "metric_direction": "maximize"},
    },
}

# ─── Task Definitions ───────────────────────────────────────────────────────

TASKS = {
    "1": {
        "label": "Language Modeling",
        "detail": "WikiText-2 + GPT-2 small (117M params)",
        "metric": "bits-per-byte (lower is better)",
        "model": {"name": "gpt2", "task": "language_modeling", "pretrained": True},
        "dataset": {
            "name": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "max_length": 512,
        },
        "agent": {"metric": "eval_loss", "metric_direction": "minimize"},
    },
    "2": {
        "label": "Text Classification",
        "detail": "SST-2 + DistilBERT",
        "metric": "accuracy (higher is better)",
        "model": {
            "name": "distilbert-base-uncased",
            "task": "text_classification",
            "pretrained": True,
        },
        "dataset": {"name": "sst2", "max_length": 128},
        "agent": {"metric": "eval_accuracy", "metric_direction": "maximize"},
    },
    "3": {
        "label": "Image Classification",
        "detail": "CIFAR-10 + ResNet-18",
        "metric": "accuracy (higher is better)",
        "model": {"name": "resnet18", "task": "image_classification", "pretrained": True},
        "dataset": {"name": "cifar10", "image_size": 32},
        "agent": {"metric": "eval_accuracy", "metric_direction": "maximize"},
    },
    "4": {
        "label": "Custom",
        "detail": "Bring your own model + dataset",
        "metric": "user-defined",
        "model": {"name": "", "task": "", "pretrained": True},
        "dataset": {"name": "custom", "custom_path": ""},
        "agent": {"metric": "eval_loss", "metric_direction": "minimize"},
    },
}

LLM_PROVIDERS = {
    "1": {
        "label": "OpenAI",
        "detail": "GPT-4o-mini — needs OPENAI_API_KEY",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "2": {
        "label": "Anthropic",
        "detail": "Claude Haiku — needs ANTHROPIC_API_KEY",
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "3": {
        "label": "Ollama (Local)",
        "detail": "Llama 3 — free, needs ~8GB RAM",
        "provider": "ollama",
        "model": "llama3",
        "env_key": None,
    },
    "4": {
        "label": "LiteLLM (Any)",
        "detail": "Any provider via LITELLM_MODEL env var",
        "provider": "litellm",
        "model": "",
        "env_key": "LITELLM_MODEL",
    },
}

TRACKERS = {
    "1": {
        "label": "Weights & Biases",
        "detail": "Cloud dashboards, free tier",
        "backend": "wandb",
        "env_key": "WANDB_API_KEY",
    },
    "2": {
        "label": "MLflow",
        "detail": "Self-hosted, no external dependency",
        "backend": "mlflow",
        "env_key": None,
    },
    "3": {
        "label": "None",
        "detail": "Local JSON logs only",
        "backend": "none",
        "env_key": None,
    },
}


def _print_banner():
    banner = Text()
    banner.append("🧬 Auto-Augment Agent", style="bold magenta")
    banner.append(" — Setup Wizard\n", style="dim")
    banner.append(
        "Autonomous Data-Augmentation & Tuning for ML models",
        style="italic cyan",
    )
    console.print(Panel(banner, border_style="magenta", padding=(1, 2)))


def _choose(title: str, options: dict, key_label: str = "Option") -> str:
    """Present a numbered choice menu and return the selected key."""
    console.print(f"\n[bold cyan]─── {title} ───[/bold cyan]")

    table = Table(show_header=False, border_style="dim", padding=(0, 2))
    table.add_column("Choice", style="bold yellow", width=4)
    table.add_column("Label", style="bold")
    table.add_column("Details", style="dim")

    for key, opt in options.items():
        table.add_row(f"[{key}]", opt["label"], opt.get("detail", ""))

    console.print(table)
    valid_keys = list(options.keys())
    choice = Prompt.ask(
        f"  Select {key_label}",
        choices=valid_keys,
        default=valid_keys[0],
    )
    return choice


def _gather_custom_config() -> dict:
    """Gather configuration for a custom task."""
    console.print("\n[bold yellow]Custom Task Configuration[/bold yellow]")

    model_name = Prompt.ask("  Model name (HuggingFace ID)", default="gpt2")
    task = Prompt.ask(
        "  Task type",
        choices=["language_modeling", "text_classification", "image_classification"],
    )
    dataset_path = Prompt.ask("  Dataset path or HuggingFace name")
    metric = Prompt.ask("  Primary metric name", default="eval_loss")
    direction = Prompt.ask(
        "  Metric direction",
        choices=["minimize", "maximize"],
        default="minimize",
    )

    return {
        "model": {"name": model_name, "task": task, "pretrained": True},
        "dataset": {"name": "custom", "custom_path": dataset_path},
        "agent": {"metric": metric, "metric_direction": direction},
    }


def run_setup_wizard(preset: str | None = None):
    """Run the interactive setup wizard or apply a preset."""
    import yaml

    from src.utils.config import load_yaml, save_yaml

    _print_banner()

    # ─── Preset mode ─────────────────────────────────────────────────────
    if preset and preset in PRESETS:
        p = PRESETS[preset]
        console.print(f"\n[green]✓ Using preset:[/green] [bold]{preset}[/bold]")
        console.print(f"  {p['description']}")
        selections = {k: v for k, v in p.items() if k != "description"}
    elif preset == "full" or preset is None:
        # ─── Interactive wizard ──────────────────────────────────────────
        selections = {}

        # 1. Task & Dataset
        task_choice = _choose("1. Target Task & Dataset", TASKS, "task")
        if task_choice == "4":
            custom = _gather_custom_config()
            selections.update(custom)
        else:
            task = TASKS[task_choice]
            selections["model"] = task["model"]
            selections["dataset"] = task["dataset"]
            selections["agent"] = task["agent"]

        # 2. LLM Provider
        llm_choice = _choose("2. LLM Provider (for config proposals)", LLM_PROVIDERS)
        llm = LLM_PROVIDERS[llm_choice]
        selections["llm"] = {
            "provider": llm["provider"],
            "model": llm["model"],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        if llm["provider"] == "litellm":
            custom_model = Prompt.ask("  LiteLLM model string", default="openai/gpt-4o-mini")
            selections["llm"]["model"] = custom_model

        # 3. Experiment Tracking
        tracker_choice = _choose("3. Experiment Tracking", TRACKERS)
        tracker = TRACKERS[tracker_choice]
        selections["tracking"] = {
            "backend": tracker["backend"],
            "project": "auto-augment-agent",
        }

        # 4. Execution Mode
        exec_choices = {
            "1": {"label": "Docker", "detail": "Isolated, reproducible — needs nvidia-container-toolkit"},
            "2": {"label": "Local", "detail": "Direct execution, simpler setup"},
        }
        exec_choice = _choose("4. Execution Mode", exec_choices)
        selections["execution"] = {
            "mode": "docker" if exec_choice == "1" else "local",
        }

        # 5. Optional Modules
        console.print("\n[bold cyan]─── 5. Optional Modules ───[/bold cyan]")
        web_fetcher = Confirm.ask("  Enable Web Fetcher (arXiv, Wikipedia)?", default=False)
        lora = Confirm.ask("  Enable LoRA fine-tuning support?", default=False)
        selections["modules"] = {"web_fetcher": web_fetcher, "lora": lora}

        # 6. Agent budget
        console.print("\n[bold cyan]─── 6. Agent Budget ───[/bold cyan]")
        max_trials = IntPrompt.ask("  Max trials to run", default=50)
        max_steps = IntPrompt.ask("  Training steps per trial (~5-10 min)", default=500)
        selections["agent"] = {
            **selections.get("agent", {}),
            "max_trials": max_trials,
        }
        selections["training"] = {"max_steps": max_steps}
    else:
        console.print(f"[red]Unknown preset: {preset}[/red]")
        console.print(f"Available presets: {', '.join(PRESETS.keys())}, full")
        sys.exit(1)

    # ─── Apply selections to base config ─────────────────────────────────
    base_config = load_yaml(CONFIG_DIR / "base.yaml")

    # Deep merge selections into base config
    for key, value in selections.items():
        if isinstance(value, dict) and key in base_config:
            base_config[key].update(value)
        else:
            base_config[key] = value

    # Save updated config
    save_yaml(base_config, CONFIG_DIR / "base.yaml")

    # ─── Create .env template ────────────────────────────────────────────
    env_lines = [
        "# Auto-Augment Agent — Environment Variables",
        "# Generated by `ada init`",
        "",
    ]

    # Add required API keys based on selections
    llm_provider = selections.get("llm", {}).get("provider", "")
    if llm_provider == "openai":
        env_lines.append("OPENAI_API_KEY=your-key-here")
    elif llm_provider == "anthropic":
        env_lines.append("ANTHROPIC_API_KEY=your-key-here")
    elif llm_provider == "litellm":
        env_lines.append(f"LITELLM_MODEL={selections['llm']['model']}")

    tracker_backend = selections.get("tracking", {}).get("backend", "none")
    if tracker_backend == "wandb":
        env_lines.append("WANDB_API_KEY=your-key-here")
    elif tracker_backend == "mlflow":
        env_lines.append("MLFLOW_TRACKING_URI=http://localhost:5000")

    env_path = PROJECT_ROOT / ".env.example"
    env_path.write_text("\n".join(env_lines) + "\n")

    # ─── Validate environment ────────────────────────────────────────────
    warnings = []

    # Check Docker if needed
    if selections.get("execution", {}).get("mode") == "docker":
        if not shutil.which("docker"):
            warnings.append("Docker not found in PATH. Install Docker to use container mode.")

    # Check GPU
    try:
        import torch

        if not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                console.print("[green]✓ Apple Silicon GPU (MPS) detected[/green]")
            else:
                warnings.append("No GPU detected. Training will use CPU (much slower).")
        else:
            console.print(f"[green]✓ GPU detected: {torch.cuda.get_device_name(0)}[/green]")
    except ImportError:
        warnings.append("PyTorch not installed. Run: pip install -e '.[all]'")

    # ─── Summary ─────────────────────────────────────────────────────────
    console.print("\n")
    summary_table = Table(title="✅ Configuration Summary", border_style="green")
    summary_table.add_column("Setting", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Task", selections.get("model", {}).get("task", "N/A"))
    summary_table.add_row("Model", selections.get("model", {}).get("name", "N/A"))
    summary_table.add_row(
        "Dataset",
        selections.get("dataset", {}).get("name", "N/A"),
    )
    summary_table.add_row(
        "LLM Provider",
        f"{selections.get('llm', {}).get('provider', 'N/A')} / "
        f"{selections.get('llm', {}).get('model', 'N/A')}",
    )
    summary_table.add_row(
        "Tracking", selections.get("tracking", {}).get("backend", "none")
    )
    summary_table.add_row(
        "Execution", selections.get("execution", {}).get("mode", "local")
    )
    summary_table.add_row(
        "Metric",
        f"{selections.get('agent', {}).get('metric', 'N/A')} "
        f"({selections.get('agent', {}).get('metric_direction', 'N/A')})",
    )

    console.print(summary_table)

    if warnings:
        console.print("\n[bold yellow]⚠ Warnings:[/bold yellow]")
        for w in warnings:
            console.print(f"  [yellow]• {w}[/yellow]")

    console.print(f"\n[green]✓ Config saved to:[/green] {CONFIG_DIR / 'base.yaml'}")
    console.print(f"[green]✓ Env template:[/green]   {env_path}")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Copy .env.example to .env and fill in API keys")
    console.print("  2. Run [cyan]ada baseline[/cyan] to establish baseline metrics")
    console.print("  3. Run [cyan]ada run[/cyan] to start the autonomous agent")


if __name__ == "__main__":
    run_setup_wizard()
