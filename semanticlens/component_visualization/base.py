"""
Abstract base class for component visualizers.

This module defines the interface that all component visualizers must implement,
providing consistent methods for analyzing neural network components across
different visualization approaches.
"""

import abc

import torch

import semanticlens as sl


class AbstractComponentVisualizer(abc.ABC):
    """
    Abstract base class for component visualization methods.

    This class defines the interface that all component visualizers must implement
    to ensure consistent behavior across different visualization approaches like
    activation-based and relevance-based methods.

    Methods
    -------
    get_act_max_sample_ids(layer_name)
        Get sample IDs of maximally activating examples for each component.
    to(device)
        Move visualizer and associated model to specified device.
    get_max_reference(*args, **kwargs)
        Get reference examples for components.
    run(*args, **kwargs)
        Run the visualization processing.

    Properties
    ----------
    layer_names : list of str
        Names of the layers being analyzed.
    metadata : dict
        Metadata about the visualizer configuration.
    """

    @abc.abstractmethod
    def get_act_max_sample_ids(self, layer_name: str) -> sl.ConceptTensor:
        """
        Get sample IDs of maximally activating examples for each component.

        Parameters
        ----------
        layer_name : str
            Name of the layer to analyze.

        Returns
        -------
        sl.ConceptTensor
            Tensor of shape (n_components, n_samples) where each row contains
            the data indices of the maximally activating samples for the
            respective component.
        """
        ...

    @property
    def layer_names(self) -> list[str]:
        """
        Get the layer names of the model being analyzed.

        Returns
        -------
        list of str
            Names of the layers in the model.
        """
        return self._layer_names

    @abc.abstractmethod
    def to(self, device: torch.device):
        """
        Move visualizer attributes and model to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device for the visualizer and associated model.
        """
        ...

    @abc.abstractmethod
    def get_max_reference(self, *args, **kwargs):
        """
        Get reference examples for specified components.

        This method should return examples (images, crops, etc.) that maximally
        activate the specified components.

        Parameters
        ----------
        *args
            Variable positional arguments specific to the visualizer.
        **kwargs
            Variable keyword arguments specific to the visualizer.

        Returns
        -------
        dict or other
            Reference examples for the specified components.
        """
        ...

    @property
    @abc.abstractmethod
    def metadata(self) -> dict:
        """
        Get metadata about the visualizer configuration.

        This metadata is used for caching and reproducibility, containing
        information about the visualizer's configuration and parameters.

        Returns
        -------
        dict
            Dictionary containing metadata about the visualizer.
        """
        ...

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the visualization processing.

        This method performs the main computation for the visualizer,
        similar to the `FeatureVisualizer` interface from zennit-crp.

        Parameters
        ----------
        *args
            Variable positional arguments for the processing.
        **kwargs
            Variable keyword arguments for the processing.
        """
        ...
