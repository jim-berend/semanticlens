"""
Abstract base class for component visualizers.

This module defines the interface that all component visualizers must implement,
providing consistent methods for analyzing neural network components across
different visualization approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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
    def run(self, *args, **kwargs) -> None:
        """Run the concept identification process on a given dataset.

        This method should process the dataset to gather the necessary
        information for identifying concepts encoded by the model-components,
        such as finding the top-activating samples for each component and caching them for later use.

        Parameters
        ----------
        *args
            Positional arguments for the analysis process, such as a `dataset`.

        **kwargs
            Additional keyword arguments for the analysis process, such as
            `batch_size` or `num_workers`.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_concept_db(self, cv: AbstractComponentVisualizer, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the concept database for a given component visualizer.

        Parameters
        ----------
        cv : AbstractComponentVisualizer
            The component visualizer to compute the concept database for.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary mapping layer names to their corresponding component embedings.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
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
