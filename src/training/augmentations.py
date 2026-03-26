"""Augmentation pipeline builder from YAML configuration.

Supports both text augmentations (NLP) and image augmentations (CV).
Builds a composable chain of transforms from the augmentations.yaml config.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Base Augmentation Interface
# =============================================================================


class Augmentation(ABC):
    """Base class for all augmentations."""

    def __init__(self, name: str, probability: float = 0.5, **kwargs):
        self.name = name
        self.probability = probability
        self.kwargs = kwargs

    def __call__(self, sample: Any) -> Any:
        if random.random() < self.probability:
            return self.apply(sample)
        return sample

    @abstractmethod
    def apply(self, sample: Any) -> Any:
        """Apply the augmentation to a sample."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability})"


# =============================================================================
# Text Augmentations
# =============================================================================


class SynonymReplacement(Augmentation):
    """Replace random words with their synonyms using WordNet or embeddings."""

    def __init__(self, probability: float = 0.1, max_replacements: int = 2, method: str = "wordnet", **kwargs):
        super().__init__("synonym_replacement", probability, **kwargs)
        self.max_replacements = max_replacements
        self.method = method
        self._wordnet = None

    def _get_synonyms(self, word: str) -> list[str]:
        if self.method == "wordnet":
            try:
                if self._wordnet is None:
                    from nltk.corpus import wordnet
                    import nltk
                    try:
                        wordnet.synsets("test")
                    except LookupError:
                        nltk.download("wordnet", quiet=True)
                        nltk.download("omw-1.4", quiet=True)
                    self._wordnet = wordnet
                synonyms = set()
                for syn in self._wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name().lower() != word.lower() and "_" not in lemma.name():
                            synonyms.add(lemma.name())
                return list(synonyms)
            except ImportError:
                logger.warning("nltk not installed, falling back to no-op")
                return []
        return []

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        n_replacements = min(self.max_replacements, len(words))
        indices = random.sample(range(len(words)), n_replacements)

        for idx in indices:
            synonyms = self._get_synonyms(words[idx])
            if synonyms:
                words[idx] = random.choice(synonyms)

        return " ".join(words)


class RandomInsertion(Augmentation):
    """Insert random synonyms of existing words at random positions."""

    def __init__(self, probability: float = 0.1, max_insertions: int = 1, **kwargs):
        super().__init__("random_insertion", probability, **kwargs)
        self.max_insertions = max_insertions
        self._syn_aug = SynonymReplacement(probability=1.0)

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(self.max_insertions):
            word = random.choice(words)
            synonyms = self._syn_aug._get_synonyms(word)
            if synonyms:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, random.choice(synonyms))

        return " ".join(words)


class RandomDeletion(Augmentation):
    """Randomly delete words from the text."""

    def __init__(self, probability: float = 0.05, **kwargs):
        super().__init__("random_deletion", probability, **kwargs)

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) <= 2:
            return text

        remaining = [w for w in words if random.random() > self.probability]
        if not remaining:
            return random.choice(words)
        return " ".join(remaining)


class RandomSwap(Augmentation):
    """Randomly swap positions of words."""

    def __init__(self, probability: float = 0.05, max_swaps: int = 1, **kwargs):
        super().__init__("random_swap", probability, **kwargs)
        self.max_swaps = max_swaps

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(self.max_swaps):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]

        return " ".join(words)


class TokenCutoff(Augmentation):
    """Remove a contiguous span of tokens."""

    def __init__(self, probability: float = 0.1, cutoff_ratio: float = 0.1, **kwargs):
        super().__init__("token_cutoff", probability, **kwargs)
        self.cutoff_ratio = cutoff_ratio

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 4:
            return text

        n_cut = max(1, int(len(words) * self.cutoff_ratio))
        start = random.randint(0, len(words) - n_cut)
        result = words[:start] + words[start + n_cut :]
        return " ".join(result) if result else text


class BackTranslation(Augmentation):
    """Paraphrase text via round-trip translation (en → pivot → en).

    Uses MarianMT models from HuggingFace for translation.
    Models are lazy-loaded on first use.
    """

    def __init__(
        self,
        probability: float = 0.15,
        source_lang: str = "en",
        pivot_lang: str = "de",
        **kwargs,
    ):
        super().__init__("back_translation", probability, **kwargs)
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self._forward_model = None
        self._backward_model = None
        self._forward_tokenizer = None
        self._backward_tokenizer = None

    def _load_models(self):
        if self._forward_model is not None:
            return

        try:
            from transformers import MarianMTModel, MarianTokenizer

            fwd_name = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.pivot_lang}"
            bwd_name = f"Helsinki-NLP/opus-mt-{self.pivot_lang}-{self.source_lang}"

            logger.info(f"Loading translation models: {fwd_name}, {bwd_name}")
            self._forward_tokenizer = MarianTokenizer.from_pretrained(fwd_name)
            self._forward_model = MarianMTModel.from_pretrained(fwd_name)
            self._backward_tokenizer = MarianTokenizer.from_pretrained(bwd_name)
            self._backward_model = MarianMTModel.from_pretrained(bwd_name)
        except Exception as e:
            logger.error(f"Failed to load translation models: {e}")
            raise

    def _translate(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def apply(self, text: str) -> str:
        self._load_models()
        try:
            pivot = self._translate(text, self._forward_model, self._forward_tokenizer)
            result = self._translate(pivot, self._backward_model, self._backward_tokenizer)
            return result
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}, returning original")
            return text


class ContextualInsertion(Augmentation):
    """Insert contextually appropriate words using a masked language model."""

    def __init__(self, probability: float = 0.1, model: str = "distilbert-base-uncased", **kwargs):
        super().__init__("contextual_insertion", probability, **kwargs)
        self.model_name = model
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline("fill-mask", model=self.model_name)

    def apply(self, text: str) -> str:
        self._load_pipeline()
        words = text.split()
        if len(words) < 3:
            return text

        insert_pos = random.randint(1, len(words) - 1)
        masked = " ".join(words[:insert_pos] + ["[MASK]"] + words[insert_pos:])

        try:
            results = self._pipeline(masked, top_k=5)
            chosen = random.choice(results)["token_str"].strip()
            words.insert(insert_pos, chosen)
            return " ".join(words)
        except Exception:
            return text


# =============================================================================
# Image Augmentations (requires torchvision)
# =============================================================================


def build_image_transforms(config: list[dict]) -> Any:
    """Build a torchvision Compose from image augmentation config.

    Returns a torchvision.transforms.Compose object.
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "torchvision required for image augmentations. "
            "Install with: pip install 'auto-augment-agent[cv]'"
        )

    transforms = []
    for aug in config:
        if not aug.get("enabled", False):
            continue

        name = aug["name"]
        prob = aug.get("probability", 0.5)

        if name == "random_horizontal_flip":
            transforms.append(T.RandomHorizontalFlip(p=prob))
        elif name == "random_crop":
            size = aug.get("size", 32)
            padding = aug.get("padding", 4)
            transforms.append(T.RandomCrop(size, padding=padding))
        elif name == "color_jitter":
            transforms.append(T.RandomApply([
                T.ColorJitter(
                    brightness=aug.get("brightness", 0.2),
                    contrast=aug.get("contrast", 0.2),
                    saturation=aug.get("saturation", 0.2),
                    hue=aug.get("hue", 0.1),
                )
            ], p=prob))
        elif name == "random_rotation":
            transforms.append(T.RandomApply([
                T.RandomRotation(degrees=aug.get("degrees", 15))
            ], p=prob))
        elif name == "random_erasing":
            transforms.append(T.RandomErasing(
                p=prob,
                scale=(aug.get("scale_min", 0.02), aug.get("scale_max", 0.33)),
            ))
        # CutMix and MixUp are handled at the batch level in train.py

    return T.Compose(transforms) if transforms else None


# =============================================================================
# Pipeline Builder
# =============================================================================

TEXT_AUGMENTATION_REGISTRY = {
    "synonym_replacement": SynonymReplacement,
    "random_insertion": RandomInsertion,
    "random_deletion": RandomDeletion,
    "random_swap": RandomSwap,
    "back_translation": BackTranslation,
    "token_cutoff": TokenCutoff,
    "contextual_insertion": ContextualInsertion,
}


class AugmentationPipeline:
    """Composable augmentation pipeline built from YAML config.

    For text tasks: applies text augmentations sequentially.
    For image tasks: returns a torchvision Compose transform.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("pipeline", {}).get("enabled", False)
        self.text_augmentations: list[Augmentation] = []
        self.image_transforms = None

        if not self.enabled:
            return

        # Build text augmentations
        for aug_config in config.get("pipeline", {}).get("text_augmentations", []):
            if not aug_config.get("enabled", False):
                continue

            name = aug_config["name"]
            if name in TEXT_AUGMENTATION_REGISTRY:
                params = {k: v for k, v in aug_config.items() if k not in ("name", "enabled")}
                aug = TEXT_AUGMENTATION_REGISTRY[name](**params)
                self.text_augmentations.append(aug)
                logger.info(f"Loaded text augmentation: {aug}")

        # Build image augmentations
        image_aug_config = config.get("pipeline", {}).get("image_augmentations", [])
        if any(a.get("enabled") for a in image_aug_config):
            self.image_transforms = build_image_transforms(image_aug_config)

    def augment_text(self, text: str) -> str:
        """Apply all enabled text augmentations to a text sample."""
        if not self.enabled or not self.text_augmentations:
            return text

        for aug in self.text_augmentations:
            text = aug(text)
        return text

    def get_image_transforms(self):
        """Return the composed image transforms, or None if not configured."""
        return self.image_transforms

    def __len__(self) -> int:
        return len(self.text_augmentations)

    def __repr__(self) -> str:
        if not self.enabled:
            return "AugmentationPipeline(disabled)"
        text_augs = [a.name for a in self.text_augmentations]
        img = "yes" if self.image_transforms else "no"
        return f"AugmentationPipeline(text={text_augs}, image_transforms={img})"
