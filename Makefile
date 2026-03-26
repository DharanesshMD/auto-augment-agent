.PHONY: help install install-dev init baseline run run-docker test lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	pip install -e .

install-dev: ## Install with all extras + dev tools
	pip install -e ".[all,dev]"

init: ## Run interactive setup wizard
	ada init

init-nlp: ## Quick setup for NLP (GPT-2 + WikiText-2)
	ada init --preset nlp-quick

init-cv: ## Quick setup for CV (ResNet-18 + CIFAR-10)
	ada init --preset cv-quick

baseline: ## Establish baseline metrics
	python scripts/create_baseline.py

run: ## Start the autonomous agent (local mode)
	python scripts/run_agent.py --no-docker

run-docker: ## Start the autonomous agent (Docker sandbox)
	python scripts/run_agent.py --docker

run-trials: ## Run N trials (default: 50)
	python scripts/run_agent.py --max-trials 50

dry-run: ## Propose configs without executing trials
	python scripts/run_agent.py --dry-run --max-trials 5

analyze: ## Analyze completed trial results
	python scripts/analyze_results.py

test: ## Run unit tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Lint code with ruff
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format: ## Auto-format code
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

docker-build: ## Build the trial runner Docker image
	docker build -t auto-augment-agent:latest .

docker-up: ## Start services with docker-compose
	docker compose up -d

docker-down: ## Stop docker-compose services
	docker compose down

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf __pycache__ src/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
