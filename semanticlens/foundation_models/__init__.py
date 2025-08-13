"""
Foundation model implementations for semantic analysis.

This module provides implementations of vision-language foundation models,
currently supporting various CLIP model variants from different sources.

Classes
-------
OpenClip
    OpenCLIP model implementation.
HF_Clip
    Hugging Face CLIP model implementation.
"""

from semanticlens.foundation_models.clip import HF_Clip, OpenClip

__all__ = [
    "OpenClip",
    "HF_Clip",
]
