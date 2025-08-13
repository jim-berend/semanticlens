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

from pathlib import Path

from semanticlens import foundation_models
from semanticlens._concept_tensor import ConceptTensor
from semanticlens.cache_setup import memory as _memory_proxy
from semanticlens.lens import Lens, label
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
    "setup_caching",
]


def setup_caching(cache_dir: str | Path = "./cache", **kwargs):
    """Enables and configures the caching system for SemanticLens.

    By default, caching is disabled. This function must be called at the
    start of a script to enable caching of expensive computations.

    Args:
        cache_dir (str or Path, optional): The directory to store cache files.
            Defaults to "./cache".
        **kwargs: Additional keyword arguments passed to joblib.Memory
                  (e.g., verbose=1 for debugging).


    # TODO add to README

    ```python
    import torch
    import semanticlens as sl
    # --- At this point, caching is completely OFF. ---
    # No files will be written to disk.

    # User decides to enable caching for their script.
    print("Enabling SemanticLens caching...")
    sl.setup_caching(cache_dir="/path/to/my/project_cache")
    # A log message will appear: "SemanticLens: Caching enabled..."

    # Now, any expensive operations will be cached automatically.
    # (Code for creating model, dataset, lens...)
    lens = sl.Lens(dataset, cv, fm, "my_dataset", "my_storage")

    # The first time this is called, it will be slow and the result will be cached.
    concept_db = lens.compute_semantic_embeddigs(layer_names)

    # The second time, it will be nearly instantaneous.
    concept_db_fast = lens.compute_semantic_embeddigs(layer_names)

    print("Done.")
    ```
    """
    _memory_proxy.configure(location=cache_dir, **kwargs)
