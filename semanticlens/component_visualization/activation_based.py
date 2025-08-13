"""
Activation-based component visualization for neural network analysis.

This module provides tools for visualizing neural network components using
activation maximization techniques, finding the input examples that most
strongly activate specific neurons or channels.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.activation_caching import ActMaxCache

MaxSamples = namedtuple("MaxSamples", ["samples", "activations"])
MaxRefs = namedtuple("MaxRefs", ["sample_ids", "activations"])

DEFAULT_STORAGE = Path("cache") / "concept_examples"

logger = logging.getLogger(__name__)


class ActivationComponentVisualizer(AbstractComponentVisualizer):
    """
    Component visualizer using activation maximization.

    This class finds and visualizes the input examples that most strongly
    activate specific neural network components using activation caching
    and maximization techniques.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to analyze.
    dataset : torch.utils.data.Dataset
        Dataset containing input examples for analysis.
    layer_names : list of str
        Names of the layers to analyze.
    storage_dir : Path, default=Path("cache")/"concept_examples"
        Directory for caching activation maxima.
    aggregation_fn : str, default="max"
        Function used for aggregating activations.
    device : torch.device or str, optional
        Device for computations. If None, uses model's device.
    num_samples : int, default=100
        Number of top activating samples to collect per component.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model being analyzed.
    dataset : torch.utils.data.Dataset
        Dataset for finding maximally activating examples.
    aggregate_fn : str
        Aggregation function for activations.
    num_samples : int
        Number of samples to collect per component.
    device : torch.device
        Device for computations.
    storage_dir : Path
        Directory for caching results.
    actmax_cache : ActMaxCache
        Cache for storing activation maxima.

    Methods
    -------
    run(batch_size=32, num_workers=None)
        Run activation maximization analysis.
    get_max_reference(concept_ids, layer_name, n_ref, batch_size=32)
        Get reference examples for specified concepts.
    get_act_max_sample_ids(layer_name)
        Get sample IDs of maximally activating examples.
    to(device)
        Move model to specified device.

    Properties
    ----------
    metadata : dict
        Metadata about the visualizer configuration.
    """

    def __init__(
        self,
        model,
        dataset,
        layer_names,
        storage_dir=Path("cache") / "concept_examples",
        aggregation_fn="max",
        device=None,
        num_samples=100,
        **kwargs,
    ):
        self.model = model
        self.dataset = dataset
        self._layer_names = layer_names
        self.aggregate_fn = aggregation_fn
        self.num_samples = num_samples
        self.device = device
        self.model.to(self.device)
        self.storage_dir = Path(storage_dir)

        self.actmax_cache = ActMaxCache(self.layer_names, n_collect=self.num_samples, aggregation_fn=self.aggregate_fn)
        self._ran = False

    def run(
        self,
        batch_size=32,
        num_workers=None,
    ):
        """
        Run activation maximization analysis on the dataset.

        Processes the entire dataset to find maximally activating examples
        for each component in the specified layers. Results are cached for
        efficient reuse.

        Parameters
        ----------
        batch_size : int, default=32
            Batch size for processing the dataset.
        num_workers : int, optional
            Number of worker processes for data loading.

        Notes
        -----
        If cached results exist and are valid, they will be loaded instead
        of recomputing. The cache is saved automatically after computation.
        """
        try:
            self.actmax_cache = ActMaxCache.load(self.storage_dir)
            logger.debug("Cache loaded from %s", self.storage_dir)
            self._ran = True
            return
        except FileNotFoundError:
            pass

        device = next(self.model.parameters()).device
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        with self.actmax_cache.hook_context(self.model):
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing dataset"):
                _ = self.model(images.to(device)).cpu()

        self.actmax_cache.store(self.storage_dir)
        logger.debug("Cache saved at ", self.storage_dir)
        self._ran = True

    def get_max_reference(self, concept_ids: int | list, layer_name: str, n_ref: int, batch_size: int = 32):
        """
        Get reference samples for specified concepts.

        Currently not implemented for ActivationComponentVisualizer.

        Parameters
        ----------
        concept_ids : int or list of int
            IDs of concepts to get references for.
        layer_name : str
            Name of the layer containing the concepts.
        n_ref : int
            Number of reference examples to retrieve.
        batch_size : int, default=32
            Batch size for processing.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented for this visualizer.
        """
        raise NotImplementedError(
            f"`get_max_reference` is not yet implemented for {self.__class__.__name__} but will be available soon."
        )
        # [ ] TODO act/grad-based cropping lxt like approach?
        # r_range = (0, n_ref) if isinstance(n_ref, int) else n_ref
        results = {}
        for i, (ids, acts) in tqdm(self.get_max(concept_ids, layer_name).items()):
            # samples = [to_pil_image(self.dataset[i][0])  if return_pil else self.dataset[i][0] for i in ids]
            samples = [to_pil_image(self.dataset[i][0]) for i in ids]
            results[i] = MaxSamples(samples, acts)

        return results

    def get_act_max_sample_ids(self, layer_name: str):
        """
        Get sample IDs of maximally activating samples for a layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer to get sample IDs for.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components, n_samples) containing the dataset
            indices of maximally activating samples for each component.
        """
        return self.actmax_cache.cache[layer_name].sample_ids

    def __repr__(self):
        """
        Return string representation of the visualizer.

        Returns
        -------
        str
            Detailed string representation including model, dataset, and configuration.
        """
        return (
            "ActBasedFeatureVisualization("
            + f"\n\tmodel={self.model.__class__.__name__},"
            + f"\n\tdataset={self.dataset.__class__.__name__},"
            + f"\n\tstorage_dir={self.storage_dir},"
            + f"\n\taggregation_fn={self.aggregate_fn},"
            + f"\n\tactmax_cache={self.actmax_cache},\n)"
        )

    def to(self, device: torch.device | str):
        """
        Move model to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device for the model.

        Returns
        -------
        ActivationComponentVisualizer
            Self for method chaining.
        """
        self.model.to(device)
        self.device = device
        return self

    @property
    def metadata(self) -> dict:
        """
        Get metadata about the visualizer configuration.

        Returns
        -------
        dict
            Dictionary containing aggregation function and cache information.
        """
        return {
            "aggregation_fn": self.aggregate_fn,
            "actmax_cache": repr(self.actmax_cache),
        }
