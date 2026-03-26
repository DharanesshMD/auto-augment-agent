"""Tests for text augmentations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.augmentations import (
    AugmentationPipeline,
    RandomDeletion,
    RandomSwap,
    TokenCutoff,
)


class TestRandomDeletion:
    def test_basic(self):
        aug = RandomDeletion(probability=1.0)
        text = "The quick brown fox jumps over the lazy dog"
        result = aug.apply(text)
        assert len(result.split()) <= len(text.split())

    def test_short_text_preserved(self):
        aug = RandomDeletion(probability=1.0)
        text = "Hi"
        result = aug.apply(text)
        assert len(result) > 0


class TestRandomSwap:
    def test_basic(self):
        aug = RandomSwap(probability=1.0, max_swaps=1)
        text = "word1 word2 word3 word4"
        result = aug.apply(text)
        # Same words, potentially different order
        assert set(result.split()) == set(text.split())

    def test_single_word(self):
        aug = RandomSwap(probability=1.0)
        text = "hello"
        result = aug.apply(text)
        assert result == text


class TestTokenCutoff:
    def test_basic(self):
        aug = TokenCutoff(probability=1.0, cutoff_ratio=0.3)
        text = "one two three four five six seven eight nine ten"
        result = aug.apply(text)
        assert len(result.split()) < len(text.split())

    def test_short_text(self):
        aug = TokenCutoff(probability=1.0)
        text = "ab"
        result = aug.apply(text)
        assert len(result) > 0


class TestAugmentationPipeline:
    def test_disabled_pipeline(self):
        config = {"pipeline": {"enabled": False}}
        pipeline = AugmentationPipeline(config)
        assert not pipeline.enabled
        assert pipeline.augment_text("hello") == "hello"

    def test_enabled_pipeline(self):
        config = {
            "pipeline": {
                "enabled": True,
                "text_augmentations": [
                    {
                        "name": "random_deletion",
                        "enabled": True,
                        "probability": 1.0,
                    }
                ],
            }
        }
        pipeline = AugmentationPipeline(config)
        assert pipeline.enabled
        assert len(pipeline.text_augmentations) == 1

    def test_no_enabled_augmentations(self):
        config = {
            "pipeline": {
                "enabled": True,
                "text_augmentations": [
                    {
                        "name": "random_deletion",
                        "enabled": False,
                        "probability": 0.1,
                    }
                ],
            }
        }
        pipeline = AugmentationPipeline(config)
        assert len(pipeline.text_augmentations) == 0

    def test_repr(self):
        config = {"pipeline": {"enabled": False}}
        pipeline = AugmentationPipeline(config)
        assert "disabled" in repr(pipeline)
