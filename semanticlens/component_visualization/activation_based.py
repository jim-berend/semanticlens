"""
Activation-based component visualization for neural network analysis.

This module provides tools for visualizing neural network components using
activation maximization techniques, finding the input examples that most
strongly activate specific neurons or channels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from semanticlens.component_visualization import aggregators
from semanticlens.component_visualization.activation_caching import ActMaxCache
from semanticlens.component_visualization.base import AbstractComponentVisualizer

logger = logging.getLogger(__name__)


class ActivationComponentVisualizer(AbstractComponentVisualizer):
    """
    Component visualizer using activation maximization.

    This class finds and visualizes the input examples that most strongly
    activate specific neural network components using activation caching
    and maximization techniques.
    """

    AGGREGATION_DEFAULTS = {
        "mean": aggregators.aggregate_conv_mean,
        "max": aggregators.aggregate_conv_max,
    }

    def __init__(
        self,
        model: nn.Module,
        dataset_model,
        dataset_fm,
        layer_names: list[str],
        num_samples: int,
        device=None,
        aggregate_fn=None,
        cache_dir: str | None = None,
    ):
        """
        Initialize the ActivationComponentVisualizer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to analyze.
        dataset_model : torch.utils.data.Dataset
            Dataset for model inference and activation collection.
        dataset_fm : torch.utils.data.Dataset
            Dataset preprocessed for foundation model encoding.
        layer_names : list of str
            Names of the layers to analyze for component visualization.
        num_samples : int
            Number of top activating samples to collect per component.
        device : torch.device or str, optional
            Device for computations. If None, uses model's device.
        aggregate_fn : callable, optional
            Function for aggregating activations. If None, uses default conv mean aggregation.
        cache_dir : str or None, optional
            Directory for caching results. If None, results will not be cached.

        Raises
        ------
        ValueError
            If any layer in layer_names is not found in the model.
        """
        self.model = model
        self.dataset = dataset_model
        self.dataset_fm = dataset_fm
        self.layer_names = layer_names
        self._check_layers()
        self._init_cache_dir(cache_dir)
        device = device or next(model.parameters()).device

        if aggregate_fn is None:
            logger.warning(f"No aggregation_fn provided using default: {aggregators.aggregate_conv_mean.__name__}")
            aggregate_fn = aggregators.aggregate_conv_mean

        self.model.to(device)

        self.actmax_cache = ActMaxCache(self.layer_names, n_collect=num_samples, aggregation_fn=aggregate_fn)

        if self.caching:
            try:
                self.actmax_cache.load(self.storage_dir)
            except FileNotFoundError:
                pass

    def _check_layers(self):
        """
        Validate that all specified layers exist in the model.

        Raises
        ------
        ValueError
            If any layer in self.layer_names is not found in the model.
        """
        for layer in self.layer_names:
            if layer not in dict(self.model.named_modules()):
                raise ValueError(f"Layer '{layer}' not found in model.")

    def _init_cache_dir(self, cache_dir):
        """
        Initialize the cache directory for storing results.

        Parameters
        ----------
        cache_dir : str or None
            Directory path for caching. If None, caching is disabled.
        """
        if cache_dir is None:
            logger.warning("No cache dir provided. Results will not be cached!")
            self._cache_root = None
        else:
            self._cache_root = Path(cache_dir)
            self._cache_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Results will be stored in {self.storage_dir}")

    @property
    def device(self):
        """
        Get the device of the model.

        Returns
        -------
        torch.device
            The device where the model parameters are located.
        """
        return next(self.model.parameters()).device

    def to(self, device):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device or str
            The target device to move the model to.

        Returns
        -------
        torch.nn.Module
            The model after being moved to the specified device.
        """
        return self.model.to(device)

    @property
    def caching(self) -> bool:
        """Check if caching is enabled."""
        return self._cache_root is not None

    @property
    def storage_dir(self):
        """
        Get the directory for storing concept visualization cache.

        Returns
        -------
        pathlib.Path
            Path to the storage directory for this visualizer instance.

        Raises
        ------
        AssertionError
            If no cache directory was provided during initialization.
        """
        assert self._cache_root, "No cache dir provided"
        return self._cache_root / self.__class__.__name__ / self.dataset.name / self.model.name

    @property
    def metadata(self) -> dict[str, str]:
        """
        Get metadata about the visualization instance.

        Returns
        -------
        dict[str, str]
            Dictionary containing metadata about the cache, dataset, and model.
        """
        return {**self.actmax_cache.metadata, "dataset": self.dataset.name, "model": self.model.name}

    def run(self, batch_size=32, num_workers=0):
        """
        Run activation maximization analysis on the dataset.

        Processes the entire dataset to find maximally activating examples
        for each component in the specified layers. Results are cached for
        efficient reuse.

        Parameters
        ----------
        batch_size : int, default=32
            Batch size for processing the dataset.
        num_workers : int, default=0
            Number of worker processes for data loading.

        Returns
        -------
        dict
            Dictionary mapping layer names to ActMax instances containing
            the top activating samples for each component.
        """
        if self._cache_root is None:
            logger.debug("No cache root provided, running computation...")
            return self._run(batch_size=batch_size, num_workers=num_workers)
        try:
            self.actmax_cache.load(self.storage_dir)
            return self.actmax_cache.cache
        except FileNotFoundError:
            logger.debug(f"Activation maximization cache not found at {self.storage_dir}. Running computation...")
            return self._run(batch_size=batch_size, num_workers=num_workers)

    @torch.no_grad()
    def _run(
        self, batch_size: int = 64, num_workers: int = 0
    ):  # -> dict[str, sl.ActMax]:# TODO other output than dict?
        """Actuall ActMax-Cache computation/population and caching."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        with self.actmax_cache.hook_context(self.model):
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing dataset"):
                _ = self.model(images.to(self.device)).cpu()

        if self._cache_root:
            self.actmax_cache.store(self.storage_dir)
            logger.debug(f"Stored activation maximization cache at {self.storage_dir}")

        return self.actmax_cache.cache

    @torch.no_grad()
    def compute_concept_db(self, lens, batch_size=32, **kwargs):
        """
        Compute the concept database for the given lens (foundation model).

        This is called from the Lens class following the Inversion of Control pattern.
        The method processes the dataset to find maximally activating samples and then
        embeds those samples using the foundation model.

        Parameters
        ----------
        lens : FoundationModel
            The foundation model used for embedding the maximally activating samples.
        batch_size : int, default=32
            Batch size for processing.
        **kwargs
            Additional keyword arguments passed to run() and embedding methods.

        Returns
        -------
        dict
            Dictionary mapping layer names to embedded concept representations.
        """
        # TODO replace lens argument with fm argument? we only require the fm atm
        self.run(batch_size=batch_size, **kwargs)

        embeds = self._embed_vision_dataset(lens, batch_size, **kwargs)

        concept_db = dict()
        for layer_name in self.layer_names:
            concept_db[layer_name] = embeds[self.get_max_reference(layer_name)]
        return concept_db

    def _embed_vision_dataset(self, fm, batch_size, **kwargs):
        """
        Embed the vision dataset using the provided foundation model.

        Parameters
        ----------
        fm : FoundationModel
            Foundation model with encode_image method for embedding images.
        batch_size : int
            Batch size for processing the dataset.
        **kwargs
            Additional keyword arguments passed to DataLoader.

        Returns
        -------
        torch.Tensor
            Tensor of shape (dataset_size, embedding_dim) containing embeddings
            for all samples in the dataset.

        Notes
        -----
        TODO: Implement caching for efficiency in repeated calls.
        """
        # TODO implement caching!
        fm.to(self.device)  # this need to also set the correct device after preprocessing!

        # temporarily replacing the dataset-transform with the fm-specific one
        # with self.dataset.apply_preprocessing(fm.preprocessing):
        dataloader = torch.utils.data.DataLoader(self.dataset_fm, batch_size=batch_size, shuffle=False, **kwargs)
        embeds = []
        with tqdm(total=len(self.dataset), desc="Embedding dataset...") as pbar_dataset:
            for batch in dataloader:
                data = batch[0] if isinstance(batch, (tuple, list)) else batch
                fm_out = fm.encode_image(data.to(self.device)).cpu()  # FIXME ensure the abc works out!
                embeds.append(fm_out)
                pbar_dataset.update(batch_size)
        embeds = torch.cat(embeds)

        assert embeds.shape[0] == len(self.dataset), "Number of embeddings does not match number of ids!"
        return embeds

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
        self._check_layer_name(layer_name)
        return self.actmax_cache.cache[layer_name].sample_ids

    def visualize_components(self, component_ids: torch.Tensor, layer_name: str, n_samples: int = 9, nrows: int = 3):
        """
        Visualize specific components by displaying their top activating samples.

        A good place to put it here since we need access to the PIL-dataset and actmax cache to implement this. However we should call a stateless function in here that abstracts complexity and can be used by other versions of the concept visualizer as well.

        Parameters
        ----------
        component_ids : torch.Tensor
            IDs of the components to visualize.
        layer_name : str
            Name of the layer containing the components.
        n_samples : int, default=9
            Number of top activating samples to display per component.
        nrows : int, default=3
            Number of rows in the grid layout for each component.
        """
        self._check_layer_name(layer_name)
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        if hasattr(self.dataset, "invert_normalization"):
            post_process = self.dataset.invert_normalization
        else:
            logger.debug("Dataset does not have invert_normalization method.")

            def post_process(x):
                return x

        pics = []
        for component_id in component_ids:
            ids = self.get_max_reference(layer_name=layer_name)[component_id][:n_samples]
            pics.append(
                make_grid(
                    [post_process(self.dataset[i][0]) for i in ids],
                    nrow=nrows,
                )
                .permute(1, 2, 0)
                .numpy()
            )
        n_pics = len(pics)

        n_cols = int(n_pics**0.5)
        n_rows = (n_pics + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

        if n_pics == 1:
            axs = [axs]
        elif n_rows == 1 or n_cols == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()

        for i, pic in enumerate(pics):
            axs[i].imshow(pic)
            axs[i].set_title(f"Neuron {component_ids[i]}")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        for i in range(n_pics, len(axs)):
            axs[i].axis("off")

        plt.suptitle(f"{self.model.name} {layer_name}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        if self._cache_root:
            component_id_str = "-".join(map(str, component_ids.tolist()))
            fdir = self.storage_dir / "plots"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / f"{layer_name}_{component_id_str}.png"
            plt.savefig(fpath)
            plt.close(fig)
            logger.info(f"Saved visualization to {fpath}")

    def _check_layer_name(self, layer_name):
        """
        Validate that a layer name exists in the configured layers.

        Parameters
        ----------
        layer_name : str
            Name of the layer to validate.

        Raises
        ------
        ValueError
            If the layer name is not found in self.layer_names.
        """
        if layer_name not in self.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found in model layers: {self.layer_names}")


# class ActivationComponentVisualizer(AbstractComponentVisualizer):
#     """
#     Component visualizer using activation maximization.

#     This class finds and visualizes the input examples that most strongly
#     activate specific neural network components using activation caching
#     and maximization techniques.

#     Parameters
#     ----------
#     model : torch.nn.Module
#         The neural network model to analyze.
#     dataset : torch.utils.data.Dataset # TODO we need a certain type of datasets!
#         Dataset containing input examples for analysis.
#     layer_names : list of str
#         Names of the layers to analyze.
#     storage_dir : Path, default=Path("cache")/"concept_examples"
#         Directory for caching activation maxima.
#     aggregation_fn : str, default="max"
#         Function used for aggregating activations.
#     device : torch.device or str, optional
#         Device for computations. If None, uses model's device.
#     num_samples : int, default=100
#         Number of top activating samples to collect per component.
#     **kwargs
#         Additional keyword arguments.

#     Attributes
#     ----------
#     model : torch.nn.Module
#         The neural network model being analyzed.
#     dataset : torch.utils.data.Dataset
#         Dataset for finding maximally activating examples.
#     aggregate_fn : str
#         Aggregation function for activations.
#     num_samples : int
#         Number of samples to collect per component.
#     device : torch.device
#         Device for computations.
#     storage_dir : Path
#         Directory for caching results.
#     actmax_cache : ActMaxCache
#         Cache for storing activation maxima.

#     Methods
#     -------
#     run(batch_size=32, num_workers=None)
#         Run activation maximization analysis.
#     get_max_reference(concept_ids, layer_name, n_ref, batch_size=32)
#         Get reference examples for specified concepts.
#     get_act_max_sample_ids(layer_name)
#         Get sample IDs of maximally activating examples.
#     to(device)
#         Move model to specified device.

#     Properties
#     ----------
#     metadata : dict
#         Metadata about the visualizer configuration.
#     """

#     def __init__(
#         self,
#         model,
#         dataset,
#         layer_names,
#         storage_dir=Path("cache") / "concept_examples",
#         aggregation_fn="max",
#         device=None,
#         num_samples=100,
#         **kwargs,
#     ):
#         self.model = model
#         self.dataset = dataset
#         self._layer_names = layer_names
#         self.aggregate_fn = aggregation_fn
#         self.num_samples = num_samples
#         self.device = device
#         self.model.to(self.device)
#         self.storage_dir = Path(storage_dir)

#         self.actmax_cache = ActMaxCache(self.layer_names, n_collect=self.num_samples, aggregation_fn=self.aggregate_fn)
#         self._ran = False

#     def run(
#         self,
#         batch_size=32,
#         num_workers=None,
#     ):
#         """
#         Run activation maximization analysis on the dataset.

#         Processes the entire dataset to find maximally activating examples
#         for each component in the specified layers. Results are cached for
#         efficient reuse.

#         Parameters
#         ----------
#         batch_size : int, default=32
#             Batch size for processing the dataset.
#         num_workers : int, optional
#             Number of worker processes for data loading.

#         Notes
#         -----
#         If cached results exist and are valid, they will be loaded instead
#         of recomputing. The cache is saved automatically after computation.
#         """
#         try:
#             self.actmax_cache = ActMaxCache.load(self.storage_dir)
#             logger.debug("Cache loaded from %s", self.storage_dir)
#             self._ran = True
#             return
#         except FileNotFoundError:
#             pass

#         device = next(self.model.parameters()).device
#         dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         )
#         with self.actmax_cache.hook_context(self.model):
#             for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing dataset"):
#                 _ = self.model(images.to(device)).cpu()

#         self.actmax_cache.store(self.storage_dir)
#         logger.debug("Cache saved at ", self.storage_dir)
#         self._ran = True

#     def get_max_reference(self, concept_ids: int | list, layer_name: str, n_ref: int, batch_size: int = 32):
#         """
#         Get reference samples for specified concepts.

#         Currently not implemented for ActivationComponentVisualizer.

#         Parameters
#         ----------
#         concept_ids : int or list of int
#             IDs of concepts to get references for.
#         layer_name : str
#             Name of the layer containing the concepts.
#         n_ref : int
#             Number of reference examples to retrieve.
#         batch_size : int, default=32
#             Batch size for processing.

#         Raises
#         ------
#         NotImplementedError
#             This method is not yet implemented for this visualizer.
#         """
#         raise NotImplementedError(
#             f"`get_max_reference` is not yet implemented for {self.__class__.__name__} but will be available soon."
#         )
#         # [ ] TODO act/grad-based cropping lxt like approach?
#         # r_range = (0, n_ref) if isinstance(n_ref, int) else n_ref
#         results = {}
#         for i, (ids, acts) in tqdm(self.get_max(concept_ids, layer_name).items()):
#             # samples = [to_pil_image(self.dataset[i][0])  if return_pil else self.dataset[i][0] for i in ids]
#             samples = [to_pil_image(self.dataset[i][0]) for i in ids]
#             results[i] = MaxSamples(samples, acts)

#         return results

#     def get_act_max_sample_ids(self, layer_name: str):
#         """
#         Get sample IDs of maximally activating samples for a layer.

#         Parameters
#         ----------
#         layer_name : str
#             Name of the layer to get sample IDs for.

#         Returns
#         -------
#         torch.Tensor
#             Tensor of shape (n_components, n_samples) containing the dataset
#             indices of maximally activating samples for each component.
#         """
#         return self.actmax_cache.cache[layer_name].sample_ids

#     def __repr__(self):
#         """
#         Return string representation of the visualizer.

#         Returns
#         -------
#         str
#             Detailed string representation including model, dataset, and configuration.
#         """
#         return (
#             "ActBasedFeatureVisualization("
#             + f"\n\tmodel={self.model.__class__.__name__},"
#             + f"\n\tdataset={self.dataset.__class__.__name__},"
#             + f"\n\tstorage_dir={self.storage_dir},"
#             + f"\n\taggregation_fn={self.aggregate_fn},"
#             + f"\n\tactmax_cache={self.actmax_cache},\n)"
#         )

#     def to(self, device: torch.device | str):
#         """
#         Move model to specified device.

#         Parameters
#         ----------
#         device : torch.device or str
#             Target device for the model.

#         Returns
#         -------
#         ActivationComponentVisualizer
#             Self for method chaining.
#         """
#         self.model.to(device)
#         self.device = device
#         return self

#     @property
#     def metadata(self) -> dict:
#         """
#         Get metadata about the visualizer configuration.

#         Returns
#         -------
#         dict
#             Dictionary containing aggregation function and cache information.
#         """
#         return {
#             "aggregation_fn": self.aggregate_fn,
#             "actmax_cache": repr(self.actmax_cache),
#         }
