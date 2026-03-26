<div align="center">

# 🧬 Auto-Augment Agent

**Autonomous Data-Augmentation & Tuning for ML Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

An autonomous agent that iteratively discovers data augmentation pipelines and hyperparameter configurations to improve your model's validation metrics — single GPU, fully reproducible.

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Configuration](#-configuration) · [Contributing](#-contributing)

</div>

---

## ✨ Features

- **🤖 Autonomous Optimization** — LLM-powered agent proposes augmentation + hyperparameter changes, trains, evaluates, and accepts/rejects automatically
- **🎯 Multi-Task Support** — Language modeling (GPT-2), text classification (DistilBERT), image classification (ResNet-18), or bring your own
- **🔒 Safe by Design** — Agent can only edit YAML configs validated against JSON schemas. PII scanning and license checks built-in
- **📊 Experiment Tracking** — Weights & Biases, MLflow, or local JSON logs
- **🐳 Docker Sandbox** — Optional isolated execution with GPU passthrough
- **🔄 Reproducible** — Deterministic seeds, config hashing, SQLite provenance database
- **🌿 Git Integration** — Accepted changes auto-create branches with config diffs and metrics
- **⚡ Single GPU** — Designed for efficient single-GPU experiments (~5-10 min per trial)

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/dharun/auto-augment-agent.git
cd auto-augment-agent
pip install -e ".[all]"
```

### Initialize

The interactive wizard configures everything:

```bash
ada init
```

Or use a preset for instant setup:

```bash
ada init --preset nlp-quick    # GPT-2 + WikiText-2
ada init --preset cv-quick     # ResNet-18 + CIFAR-10
```

### Run

```bash
# 1. Establish baseline metrics
ada baseline

# 2. Start the autonomous agent
ada run --max-trials 50

# 3. Analyze results
ada analyze
```

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Loop                            │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │  Propose  │───▶│ Validate │───▶│  Train   │           │
│  │  (LLM)   │    │ (Schema) │    │ (PyTorch)│           │
│  └──────────┘    └──────────┘    └──────────┘           │
│       ▲                               │                  │
│       │                               ▼                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │  Update   │◀──│  Decide  │◀──│ Evaluate │           │
│  │  History  │    │ (Gate)   │    │ (Metrics)│           │
│  └──────────┘    └──────────┘    └──────────┘           │
│                       │                                  │
│                       ▼                                  │
│              ┌────────────────┐                          │
│              │ Accept/Reject  │                          │
│              │ Git Branch/PR  │                          │
│              └────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

1. **Propose**: LLM analyzes trial history and proposes YAML config changes
2. **Validate**: Changes validated against JSON schemas (safe parameter ranges)
3. **Train**: Model trains with proposed augmentations + hyperparameters
4. **Evaluate**: Compute validation metrics (loss, BPB, accuracy, perplexity)
5. **Decide**: Accept if improvement ≥ threshold; reject otherwise
6. **Provenance**: Accepted changes create git branches with diffs and metrics

## ⚙️ Configuration

The agent can only modify two files (validated against schemas):

### `config/augmentations.yaml`
| Augmentation | Type | Description |
|---|---|---|
| `synonym_replacement` | Text | Replace words with WordNet synonyms |
| `random_insertion` | Text | Insert random synonyms |
| `random_deletion` | Text | Randomly remove words |
| `random_swap` | Text | Swap word positions |
| `back_translation` | Text | Round-trip translation for paraphrasing |
| `token_cutoff` | Text | Remove contiguous token spans |
| `contextual_insertion` | Text | Insert words via masked LM |
| `random_horizontal_flip` | Image | Horizontal flip |
| `random_crop` | Image | Random crop with padding |
| `color_jitter` | Image | Random color adjustments |
| `random_rotation` | Image | Random rotation |
| `random_erasing` | Image | Random patch erasing |
| `cutmix` / `mixup` | Image | Sample mixing strategies |

### `config/tuning.yaml`
| Parameter | Range | Description |
|---|---|---|
| `learning_rate` | `[1e-7, 1e-2]` | Optimizer learning rate |
| `optimizer` | `adamw, adam, sgd, adafactor` | Optimizer type |
| `scheduler` | `cosine, linear, constant` | LR scheduler |
| `train_batch_size` | `{1,2,4,8,16,32,64}` | Training batch size |
| `dropout` | `[0, 0.5]` | Regularization dropout |
| `lora.rank` | `{4,8,16,32,64}` | LoRA adapter rank |
| `label_smoothing` | `[0, 0.3]` | Label smoothing factor |

## 🔌 LLM Providers

| Provider | Model | Setup |
|---|---|---|
| **OpenAI** | GPT-4o-mini | Set `OPENAI_API_KEY` |
| **Anthropic** | Claude Haiku | Set `ANTHROPIC_API_KEY` |
| **Ollama** | Llama 3 (local) | Install Ollama, no API key needed |
| **LiteLLM** | Any supported model | Set `LITELLM_MODEL` |

## 📊 Experiment Tracking

| Backend | Setup | Dashboard |
|---|---|---|
| **W&B** | Set `WANDB_API_KEY` | wandb.ai |
| **MLflow** | `docker compose --profile mlflow up` | localhost:5000 |
| **None** | Default | Local JSON in `logs/` |

## 🐳 Docker

```bash
# Build the trial runner image
make docker-build

# Run with Docker isolation
ada run --docker --max-trials 50

# Start MLflow alongside
docker compose --profile mlflow up -d
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[all,dev]"

# Run tests
make test

# Lint & format
make lint
make format
```

## 📁 Project Structure

```
auto-augment-agent/
├── config/                    # YAML configs + JSON schemas
│   ├── base.yaml             # Immutable project config
│   ├── augmentations.yaml    # Agent-editable augmentations
│   ├── tuning.yaml           # Agent-editable hyperparameters
│   └── schema/               # JSON Schema constraints
├── src/
│   ├── agent/                # Proposer, evaluator, runner
│   ├── training/             # Train loop, augmentations, LoRA, metrics
│   ├── data/                 # Dataset loaders, web fetcher
│   ├── safety/               # PII scanner, license checker
│   ├── tracking/             # W&B/MLflow/JSON tracker, provenance DB
│   └── utils/                # Config, reproducibility, Docker
├── scripts/                  # CLI entry points
├── tests/                    # Unit tests
├── Dockerfile                # Trial sandbox
├── docker-compose.yml        # GPU-enabled services
└── .github/workflows/        # CI/CD pipeline
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Submit a PR

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
