FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Auto-Augment Agent"
LABEL description="Sandboxed trial runner for autonomous augmentation experiments"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source code
COPY . .

# Default entry point: run a single trial
ENTRYPOINT ["python", "scripts/run_single_trial.py"]
CMD ["--help"]
