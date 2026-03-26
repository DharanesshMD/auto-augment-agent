"""License compatibility checker for datasets and models."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Permissive licenses compatible with fine-tuning and augmentation
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "unlicense",
    "openrail",
    "openrail++",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "llama2",
    "llama3",
    "gemma",
}

# Licenses that restrict commercial use but allow research
RESEARCH_ONLY_LICENSES = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-4.0",
}


class LicenseChecker:
    """Verify license compatibility for datasets and models."""

    def check(self, config: dict[str, Any]) -> dict[str, Any]:
        """Check license compatibility of the configured dataset and model.

        Args:
            config: Full merged configuration.

        Returns:
            Dict with 'passed' (bool) and 'issues' (list of warnings).
        """
        issues = []
        model_name = config.get("model", {}).get("name", "")
        dataset_name = config.get("dataset", {}).get("name", "")

        # Check model license
        model_license = self._get_hf_license(model_name, "model")
        if model_license:
            if not self._is_compatible(model_license):
                issues.append(
                    f"Model '{model_name}' has license '{model_license}' "
                    f"which may restrict fine-tuning."
                )

        # Check dataset license
        dataset_license = self._get_hf_license(dataset_name, "dataset")
        if dataset_license:
            if not self._is_compatible(dataset_license):
                issues.append(
                    f"Dataset '{dataset_name}' has license '{dataset_license}' "
                    f"which may restrict augmentation/modification."
                )

        # Check web fetcher sources
        if config.get("modules", {}).get("web_fetcher", False):
            issues.append(
                "Web fetcher is enabled. Ensure fetched data sources "
                "have compatible licenses before using in training."
            )

        return {
            "passed": len(issues) == 0,
            "model_license": model_license,
            "dataset_license": dataset_license,
            "issues": issues,
        }

    def _get_hf_license(self, name: str, resource_type: str) -> str | None:
        """Try to fetch license info from HuggingFace Hub."""
        if not name or name in ("custom", "cifar10", "cifar100"):
            return None

        try:
            from huggingface_hub import model_info, dataset_info

            if resource_type == "model":
                info = model_info(name)
            else:
                info = dataset_info(name)

            license_tag = getattr(info, "license", None) or getattr(
                info, "card_data", {}
            )
            if isinstance(license_tag, str):
                return license_tag.lower()
            return None
        except Exception:
            logger.debug(f"Could not fetch license for {name}")
            return None

    def _is_compatible(self, license_str: str) -> bool:
        """Check if a license is compatible with augmentation + fine-tuning."""
        normalized = license_str.lower().strip()
        return (
            normalized in PERMISSIVE_LICENSES
            or normalized in RESEARCH_ONLY_LICENSES
        )
