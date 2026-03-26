"""PII (Personally Identifiable Information) scanner for augmented data.

Uses Microsoft Presidio to detect PII in generated/augmented training data.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PIIScanner:
    """Scans augmented data samples for PII before training."""

    ENTITY_TYPES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "IBAN_CODE",
    ]

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._analyzer = None

    @property
    def analyzer(self):
        """Lazy-load the Presidio analyzer."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine

                self._analyzer = AnalyzerEngine()
                logger.info("Presidio PII analyzer loaded")
            except ImportError:
                raise ImportError(
                    "presidio-analyzer required for PII scanning. "
                    "Install with: pip install presidio-analyzer presidio-anonymizer"
                )
        return self._analyzer

    def scan_text(self, text: str) -> list[dict[str, Any]]:
        """Scan a text sample for PII.

        Returns list of detected PII entities.
        """
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.ENTITY_TYPES,
            score_threshold=self.threshold,
        )
        return [
            {
                "type": r.entity_type,
                "score": r.score,
                "start": r.start,
                "end": r.end,
                "text": text[r.start : r.end],
            }
            for r in results
        ]

    def scan_batch(self, texts: list[str], sample_size: int = 100) -> dict[str, Any]:
        """Scan a batch of texts, sampling if the batch is large.

        Args:
            texts: List of text samples to scan.
            sample_size: Max number of samples to scan (for efficiency).

        Returns:
            Dict with 'safe' (bool), 'total_scanned', 'pii_found' (list).
        """
        import random

        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)

        all_pii = []
        for text in texts:
            pii = self.scan_text(text)
            if pii:
                all_pii.extend(pii)

        return {
            "safe": len(all_pii) == 0,
            "total_scanned": len(texts),
            "pii_count": len(all_pii),
            "pii_found": all_pii[:10],  # Limit output
            "issues": [
                f"Found {p['type']} (score={p['score']:.2f}): '{p['text']}'"
                for p in all_pii[:5]
            ],
        }

    def check_config(self, aug_config: dict) -> dict[str, Any]:
        """Quick check if augmentation config has any PII-risky settings.

        For example, if data_mixing is enabled with external sources.
        """
        pipeline = aug_config.get("pipeline", {})
        issues = []

        # Check data mixing sources
        mixing = pipeline.get("data_mixing", {})
        if mixing.get("enabled", False) and mixing.get("sources"):
            issues.append(
                "Data mixing with external sources is enabled. "
                "PII scanning will run on mixed data during training."
            )

        return {
            "safe": len(issues) == 0,
            "issues": issues,
        }
