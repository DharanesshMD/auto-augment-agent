"""Docker container management for isolated trial execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DockerTrialRunner:
    """Runs training trials in isolated Docker containers with GPU passthrough."""

    def __init__(
        self,
        image: str = "auto-augment-agent:latest",
        gpu_ids: str = "0",
        memory_limit: str = "16g",
        network_enabled: bool = False,
    ):
        self.image = image
        self.gpu_ids = gpu_ids
        self.memory_limit = memory_limit
        self.network_enabled = network_enabled
        self._client = None

    @property
    def client(self):
        """Lazy-load Docker client."""
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
                self._client.ping()
            except ImportError:
                raise RuntimeError(
                    "Docker Python SDK not installed. "
                    "Install it with: pip install 'auto-augment-agent[docker]'"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to Docker daemon: {e}\n"
                    "Make sure Docker is running."
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Docker is available and the image exists."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def build_image(self, context_path: str | Path = ".") -> str:
        """Build the trial runner Docker image."""
        logger.info(f"Building Docker image: {self.image}")
        image, logs = self.client.images.build(
            path=str(context_path),
            tag=self.image,
            rm=True,
        )
        for chunk in logs:
            if "stream" in chunk:
                logger.debug(chunk["stream"].strip())
        return image.id

    def run_trial(
        self,
        trial_id: int,
        config_dir: str | Path,
        output_dir: str | Path,
        data_dir: str | Path | None = None,
        timeout: int = 900,  # 15 min default timeout
    ) -> dict[str, Any]:
        """Run a single trial in a Docker container.

        Args:
            trial_id: Unique trial identifier.
            config_dir: Path to config directory (mounted read-only).
            output_dir: Path to output directory (mounted read-write).
            data_dir: Optional path to data directory (mounted read-only).
            timeout: Maximum runtime in seconds.

        Returns:
            Dictionary with trial results (metrics, status, logs).
        """
        config_dir = Path(config_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        volumes = {
            str(config_dir): {"bind": "/app/config", "mode": "ro"},
            str(output_dir): {"bind": "/app/outputs", "mode": "rw"},
        }

        if data_dir:
            data_dir = Path(data_dir).resolve()
            volumes[str(data_dir)] = {"bind": "/app/data_cache", "mode": "ro"}

        # GPU configuration
        device_requests = []
        if self.gpu_ids:
            device_requests = [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": self.gpu_ids.split(","),
                    "Capabilities": [["gpu"]],
                }
            ]

        container_name = f"ada-trial-{trial_id}"
        logger.info(f"Starting trial {trial_id} in container: {container_name}")

        try:
            container = self.client.containers.run(
                image=self.image,
                command=["--trial-id", str(trial_id)],
                name=container_name,
                volumes=volumes,
                device_requests=device_requests if device_requests else None,
                mem_limit=self.memory_limit,
                network_mode="none" if not self.network_enabled else "bridge",
                environment={
                    "TRIAL_ID": str(trial_id),
                    "CUDA_VISIBLE_DEVICES": self.gpu_ids,
                },
                detach=True,
                remove=False,
            )

            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            exit_code = result.get("StatusCode", -1)

            # Read results from output directory
            results_file = output_dir / f"trial_{trial_id}_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    trial_results = json.load(f)
            else:
                trial_results = {}

            trial_results.update({
                "trial_id": trial_id,
                "exit_code": exit_code,
                "container_logs": logs[-5000:],  # Last 5K chars of logs
                "status": "success" if exit_code == 0 else "failed",
            })

            return trial_results

        except Exception as e:
            logger.error(f"Trial {trial_id} container error: {e}")
            return {
                "trial_id": trial_id,
                "status": "error",
                "error": str(e),
            }
        finally:
            # Cleanup container
            try:
                container = self.client.containers.get(container_name)
                container.remove(force=True)
            except Exception:
                pass

    def cleanup(self) -> None:
        """Remove all trial containers and dangling images."""
        try:
            containers = self.client.containers.list(
                all=True,
                filters={"name": "ada-trial-"},
            )
            for c in containers:
                c.remove(force=True)
                logger.info(f"Removed container: {c.name}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
