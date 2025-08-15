"""
Abstract base class for component visualizers.

This module defines the interface that all component visualizers must implement,
providing consistent methods for analyzing neural network components across
different visualization approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class AbstractComponentVisualizer(ABC):
    """
    Abstract base class for all component visualizers.

    A component visualizer is responsible for identifying the "concepts"
    that a model's components (e.g., neurons, channels) have learned.
    This is typically done by analyzing how the components respond to a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to analyze.
    device : str or torch.device, optional
        Device for computations. If None, uses the model's current device.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model being analyzed.
    device : torch.device
        The device where computations are performed.
    """

    def __init__(self, model: torch.nn.Module, device: str | torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @abstractmethod
    def run(self, dataset: Dataset, **kwargs) -> None:
        """
        Run the concept identification process on a given dataset.

        This method should process the dataset to gather the necessary
        information for identifying concepts (e.g., top-activating samples).

        Parameters
        ----------
        dataset : Dataset
            The dataset to process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_reference(self, layer_name) -> torch.Tensor:
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
        raise NotImplementedError

    def to(self, device: str | torch.device):
        """
        Move the visualizer and its model to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The target device to move the model to.

        Returns
        -------
        AbstractComponentVisualizer
            Returns self for method chaining.
        """
        self.device = device
        self.model.to(self.device)
        return self

    @property
    def metadata(self) -> dict[str, str]:
        """
        Get metadata about the visualization instance.

        Returns
        -------
        dict[str, str]
            Dictionary containing metadata about the visualizer.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def caching(self) -> bool:
        """
        Check if caching is enabled.

        Returns
        -------
        bool
            True if caching is enabled, False otherwise.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def storage_dir(self):
        """
        Get the directory for storing concept visualization cache.

        Returns
        -------
        pathlib.Path
            Path to the directory where cache files are stored.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def device(self):
        """
        Get the device on which the model is located.

        Returns
        -------
        torch.device
            The device where the model parameters are located.
        """
        return next(self.model.parameters()).device


# class AbstractComponentVisualizer(abc.ABC):
#     """
#     Abstract base class for component visualization methods.

#     This class defines the interface that all component visualizers must implement
#     to ensure consistent behavior across different visualization approaches like
#     activation-based and relevance-based methods.

#     Methods
#     -------
#     get_act_max_sample_ids(layer_name)
#         Get sample IDs of maximally activating examples for each component.
#     to(device)
#         Move visualizer and associated model to specified device.
#     get_max_reference(*args, **kwargs)
#         Get reference examples for components.
#     run(*args, **kwargs)
#         Run the visualization processing.

#     Properties
#     ----------
#     layer_names : list of str
#         Names of the layers being analyzed.
#     metadata : dict
#         Metadata about the visualizer configuration.
#     """

#     @abc.abstractmethod
#     def get_act_max_sample_ids(self, layer_name: str) -> sl.ConceptTensor:
#         """
#         Get sample IDs of maximally activating examples for each component.

#         Parameters
#         ----------
#         layer_name : str
#             Name of the layer to analyze.

#         Returns
#         -------
#         sl.ConceptTensor
#             Tensor of shape (n_components, n_samples) where each row contains
#             the data indices of the maximally activating samples for the
#             respective component.
#         """
#         ...

#     @property
#     def layer_names(self) -> list[str]:
#         """
#         Get the layer names of the model being analyzed.

#         Returns
#         -------
#         list of str
#             Names of the layers in the model.
#         """
#         return self._layer_names

#     @abc.abstractmethod
#     def to(self, device: torch.device):
#         """
#         Move visualizer attributes and model to the specified device.

#         Parameters
#         ----------
#         device : torch.device or str
#             Target device for the visualizer and associated model.
#         """
#         ...

#     @abc.abstractmethod
#     def get_max_reference(self, *args, **kwargs):
#         """
#         Get reference examples for specified components.

#         This method should return examples (images, crops, etc.) that maximally
#         activate the specified components.

#         Parameters
#         ----------
#         *args
#             Variable positional arguments specific to the visualizer.
#         **kwargs
#             Variable keyword arguments specific to the visualizer.

#         Returns
#         -------
#         dict or other
#             Reference examples for the specified components.
#         """
#         ...

#     @property
#     @abc.abstractmethod
#     def metadata(self) -> dict:
#         """
#         Get metadata about the visualizer configuration.

#         This metadata is used for caching and reproducibility, containing
#         information about the visualizer's configuration and parameters.

#         Returns
#         -------
#         dict
#             Dictionary containing metadata about the visualizer.
#         """
#         ...

#     @abc.abstractmethod
#     def run(self, *args, **kwargs):
#         """
#         Run the visualization processing.

#         This method performs the main computation for the visualizer,
#         similar to the `FeatureVisualizer` interface from zennit-crp.

#         Parameters
#         ----------
#         *args
#             Variable positional arguments for the processing.
#         **kwargs
#             Variable keyword arguments for the processing.
#         """
#         ...
