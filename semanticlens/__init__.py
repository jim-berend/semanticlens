"""
SemanticLens: A package for mechanistic understanding and validation of large AI models.

SemanticLens provides tools for visual concept analysis and exploration of deep learning models,
specifically designed for mechanistic interpretability and semantic analysis of foundation models.

Modules
-------
foundation_models
    Contains foundation model implementations including CLIP variants.
scores
    Provides scoring functions for concept clarity, redundancy, and polysemanticity.

Classes
-------
ConceptTensor
    A tensor subclass for storing embeddings with associated metadata.
Lens
    Main class for visual concept analysis and exploration.

Functions
---------
label
    Compute alignment of text embeddings with concept embeddings.
clarity_score
    Measure how uniform concept examples are.
polysemanticity_score
    Measure concept polysemanticity using clustering.
redundancy_score
    Measure concept redundancy across neurons.
"""

from __future__ import annotations

from semanticlens import foundation_models
from semanticlens.lens import Lens
from semanticlens.scores import clarity_score, polysemanticity_score, redundancy_score

from . import scores

__all__ = [
    "scores",
    "ConceptTensor",
    "foundation_models",
    "Lens",
    "label",
    "clarity_score",
    "polysemanticity_score",
    "redundancy_score",
]
