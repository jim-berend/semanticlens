import abc

import torch

import semanticlens as sl


class AbstractComponentVisualizer(abc.ABC):
    @abc.abstractmethod
    def get_act_max_sample_ids(self, layer_name: str) -> sl.ConceptTensor:
        """
        Should return a tensor of shape (n_components, n_samples) for the given layer name.
        Where each row holds the data indices of the maximaly activating samples for the repective component.
        """
        ...

    @property
    def layer_names(self) -> list[str]:
        """Returns the layer names of the model"""
        return self._layer_names

    @abc.abstractmethod
    def to(self, device: torch.device):
        """Moves attributes and model to the given device"""
        ...

    @abc.abstractmethod
    def get_max_reference(self, *args, **kwargs): ...

    @property
    @abc.abstractmethod
    def metadata(self) -> dict:
        """
        Returns the metadata of the visualizer.
        Required for caching
        """
        ...

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """
        Runs visualizer processing cf. zennit-crp `FeatureVisualizer`.
        """
        ...
