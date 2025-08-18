"""Helper functions for SemanticLens."""

from __future__ import annotations

import hashlib

import torch
from torchvision import transforms


def _string_hash(s: str) -> int:
    """Generate a stable hash for a given string."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def get_fallback_name(obj):
    """Function to get a fallback name for an object."""
    return obj.__class__.__name__ + "-" + str(_string_hash(str(obj)))


def to_transforms_compose(instance: transforms._presets.ImageClassification) -> transforms.Compose:
    """Helper function to convert an ImageClassification instance to a Compose transform."""
    return transforms.Compose(
        [
            transforms.Resize(
                instance.resize_size,
                interpolation=instance.interpolation,
                antialias=instance.antialias,
            ),
            transforms.CenterCrop(instance.crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=instance.mean, std=instance.std),
        ]
    )


def get_unnormalization_transform(
    mean: torch.Tensor | list[float] = torch.tensor([0.485, 0.456, 0.406]),
    std: torch.Tensor | list[float] = torch.tensor([0.229, 0.224, 0.225]),
) -> torch.nn.Module:
    """Return a transform to undo the normalization of an image tensor.

    This function creates a composition of transforms that reverses standard
    ImageNet normalization, which is useful for visualization purposes.

    Parameters
    ----------
    mean : torch.Tensor, optional
        Mean values used in the original normalization. Default is ImageNet means
        [0.485, 0.456, 0.406] for RGB channels.
    std : torch.Tensor, optional
        Standard deviation values used in the original normalization. Default is
        ImageNet stds [0.229, 0.224, 0.225] for RGB channels.

    Returns
    -------
    torch.nn.Module
        A composed transform that reverses the normalization when applied to a tensor.
    """
    from torchvision import transforms

    if isinstance(mean, list):
        mean = torch.tensor(mean, dtype=torch.float32)

    if isinstance(std, list):
        std = torch.tensor(std, dtype=torch.float32)

    return transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )
