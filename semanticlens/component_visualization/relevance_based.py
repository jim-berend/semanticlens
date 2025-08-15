"""
Relevance-based component visualization using attribution methods.

This module provides tools for visualizing neural network components using
Layer-wise Relevance Propagation (LRP) and Concept Relevance Propagation (CRP)
attribution methods to understand which input features are most relevant for
specific neural activations.
"""

from __future__ import annotations

import logging
import os

import torch
from crp.concepts import ChannelConcept as Concept
from crp.helper import load_maximization
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat as Composite

from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.render import crop_and_mask_images

logger = logging.getLogger(__name__)


# NOTE this is currently broken and under heavy development! FIXME


class RelevanceComponentVisualizer(FeatureVisualization, AbstractComponentVisualizer):
    """
    Component visualizer using relevance-based attribution methods.

    This class extends the FeatureVisualization from CRP (Concept Relevance Propagation)
    to provide relevance-based analysis of neural network components. It uses attribution
    methods like LRP to understand which input features contribute most to specific
    neural activations.

    Parameters
    ----------
    attribution : crp.attribution.Attributor
        Attribution method for computing relevance scores.
    dataset : torch.utils.data.Dataset
        Dataset containing input examples for analysis.
    layer_names : str or list of str
        Names of the layers to analyze.
    preprocess_fn : callable
        Function for preprocessing input data.
    composite : zennit.composites.Composite, optional
        Composite rule for attribution computation.
    aggregation_fn : str, default="sum"
        Function for aggregating relevance scores.
    abs_norm : bool, default=True
        Whether to use absolute normalization.
    storage_dir : str, default="FeatureVisualization"
        Directory for storing visualization results.
    device : torch.device or str, optional
        Device for computations.
    num_samples : int, default=100
        Number of samples to analyze per component.
    cache : optional
        Caching configuration.
    plot_fn : callable, default=crop_and_mask_images
        Function for plotting/rendering visualizations.

    Attributes
    ----------
    num_samples : int
        Number of samples per component.
    composite : zennit.composites.Composite
        Composite rule for attribution.
    storage_dir : str
        Storage directory path.
    plot_fn : callable
        Plotting function.
    aggregation_fn : str
        Aggregation function name.
    abs_norm : bool
        Whether absolute normalization is used.
    ActMax : crp.maximization.Maximization
        Maximization handler for activation analysis.
    ActStats : crp.statistics.Statistics
        Statistics handler for activation analysis.

    Methods
    -------
    run(composite=None, data_start=0, data_end=None, batch_size=32, checkpoint=500, on_device=None, **kwargs)
        Run relevance-based preprocessing and analysis.
    get_max_reference(concept_ids, layer_name, n_ref, batch_size=32)
        Get reference examples using relevance attribution.
    check_if_preprocessed()
        Check if preprocessing has been completed.
    get_act_max_sample_ids(layer_name)
        Get sample IDs of maximally activating examples.
    to(device)
        Move visualizer to specified device.

    Properties
    ----------
    metadata : dict
        Metadata about the visualizer configuration.
    """

    def __init__(
        self,
        attribution,
        dataset,
        layer_names,
        preprocess_fn,
        composite=None,
        aggregation_fn="sum",
        abs_norm=True,
        storage_dir="FeatureVisualization",
        device=None,
        num_samples=100,
        cache=None,
        plot_fn=crop_and_mask_images,
    ):
        layer_names = [layer_names] if not isinstance(layer_names, list) else layer_names
        super().__init__(
            attribution,
            dataset,
            layer_map={layer_name: Concept() for layer_name in layer_names},
            preprocess_fn=preprocess_fn,
            max_target=aggregation_fn,
            abs_norm=abs_norm,
            path=storage_dir,
            device=device,
            cache=cache,
        )
        self._layer_names = layer_names

        self.num_samples = num_samples
        self.composite = composite
        self.storage_dir = storage_dir
        self.plot_fn = plot_fn
        self.aggregation_fn = aggregation_fn
        self.abs_norm = abs_norm

        from crp.maximization import Maximization
        from crp.statistics import Statistics

        # set normalization to false
        self.ActMax = Maximization(mode="activation", max_target=aggregation_fn, abs_norm=False, path=self.storage_dir)
        self.ActStats = Statistics(mode="activation", max_target=aggregation_fn, abs_norm=False, path=self.storage_dir)

        self.ActMax.SAMPLE_SIZE = self.num_samples

        self._ran = self.check_if_preprocessed()

    def run(
        self,
        composite: Composite = None,
        data_start=0,
        data_end=None,
        batch_size=32,
        checkpoint=500,
        on_device=None,
        **kwargs,
    ):
        """
        Run relevance-based preprocessing and analysis.

        Processes the dataset using attribution methods to compute relevance
        scores and identify maximally activating examples for each component.

        Parameters
        ----------
        composite : zennit.composites.Composite, optional
            Composite rule for attribution computation. If None, uses the
            composite from initialization.
        data_start : int, default=0
            Starting index in the dataset.
        data_end : int, optional
            Ending index in the dataset. If None, processes entire dataset.
        batch_size : int, default=32
            Batch size for processing.
        checkpoint : int, default=500
            Interval for saving checkpoints during processing.
        on_device : torch.device or str, optional
            Device for computation.
        **kwargs
            Additional keyword arguments passed to parent run method.

        Returns
        -------
        list or other
            Results from preprocessing, or list of existing files if already preprocessed.
        """
        composite = self._check_composite(composite)
        if not self.check_if_preprocessed():
            logging.info("Preprocessing...")
            data_end = len(self.dataset) if data_end is None else data_end
            results = super().run(composite, data_start, data_end, batch_size, checkpoint, on_device)
            self._ran = True
            return results

        else:
            logging.info("Already preprocessed")
            return [j for j in os.listdir(self.ActMax.PATH) if any([layer in j for layer in self.layer_names])]

    @torch.enable_grad()  # required for LRP/CRP
    def get_max_reference(self, concept_ids: int | list, layer_name: str, n_ref: int, batch_size: int = 32):
        """
        Get reference examples using relevance attribution.

        Computes relevance-based visualizations for specified concepts using
        attribution methods to highlight the most relevant input features.

        Parameters
        ----------
        concept_ids : int or list of int
            IDs of concepts to visualize.
        layer_name : str
            Name of the layer containing the concepts.
        n_ref : int
            Number of reference examples to generate.
        batch_size : int, default=32
            Batch size for processing.

        Returns
        -------
        dict
            Dictionary mapping concept IDs to their reference visualizations.

        Raises
        ------
        AttributeError
            If gradients are not enabled or CRP requirements are not met.

        Notes
        -----
        This method requires gradients to be enabled for LRP/CRP computation.
        The `torch.enable_grad()` decorator ensures this requirement is met.
        """
        mode = "activation"
        r_range = (0, n_ref)
        composite = self.composite
        plot_fn = self.plot_fn
        rf = True
        try:
            return super().get_max_reference(concept_ids, layer_name, mode, r_range, composite, rf, plot_fn, batch_size)
        except AttributeError as e:
            logging.error("Error during LRP/CRP-based concept-visualization.")
            logging.error("Note `crp` requires gradients: Make sure to execute with torch autograd enabled.")
            raise e

    def check_if_preprocessed(self):
        """
        Check if preprocessing has been completed for all layers.

        Returns
        -------
        bool
            True if all specified layers have been preprocessed, False otherwise.
        """
        return bool(os.listdir(self.ActMax.PATH)) and all(
            any([i.startswith(layer_name) for i in os.listdir(self.ActMax.PATH)]) for layer_name in self.layer_names
        )

    def _check_composite(self, composite):
        """
        Validate and return the composite rule to use.

        Parameters
        ----------
        composite : zennit.composites.Composite or None
            Composite rule provided to method.

        Returns
        -------
        zennit.composites.Composite
            The composite rule to use for attribution.

        Raises
        ------
        AssertionError
            If no composite is provided and none was set during initialization.
        """
        assert composite or self.composite, "Composite must be provided or set in initialization (__init__)"
        return composite or self.composite

    def get_act_max_sample_ids(self, layer_name: str):
        """
        Get sample IDs of maximally activating examples for a layer.

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
        return torch.tensor(load_maximization(path_folder=self.ActMax.PATH, layer_name=layer_name)[0]).T

    def to(self, device: torch.device | str):
        """
        Move visualizer and attribution model to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device for the visualizer and attribution model.
        """
        self.device = device
        self.attribution.model.to(self.device)

    @property
    def metadata(self) -> dict:
        """
        Get metadata about the visualizer configuration.

        Returns
        -------
        dict
            Dictionary containing configuration parameters for caching and
            reproducibility, including preprocessing function, normalization
            settings, aggregation function, storage directory, device,
            number of samples, and plotting function.
        """
        return {
            "preprocess_fn": str(self.preprocess_fn),
            "abs_norm": self.abs_norm,
            "aggregation_fn": self.aggregation_fn,
            "storage_dir": str(self.storage_dir),
            "device": str(self.device),
            "num_samples": self.num_samples,
            "plot_fn": str(self.plot_fn),
        }
