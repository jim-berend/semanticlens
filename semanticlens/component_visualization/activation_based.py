"""Activation-based component visualization for neural network analysis.

This module provides the `ActivationComponentVisualizer`, a tool for identifying
the concept examples that most strongly activate specific components (e.g., neurons,
channels) of a neural network. It works by performing a forward pass over a dataset,
caching the activations for specified layers, and identifying the top-k activating
input samples for each component.
For component visualization the full act-max samples are returned.

"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from semanticlens.component_visualization import aggregators
from semanticlens.component_visualization.activation_caching import ActMaxCache
from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.helper import get_fallback_name

logger = logging.getLogger(__name__)


class MissingNameWarning(UserWarning):
    """
    Warning raised when a model or dataset is missing a `.name` attribute.

    This attribute is crucial for the caching mechanism to create a stable and predictable cache location.
    Without it, a fallback name is generated.
    """

    pass


class ActivationComponentVisualizer(AbstractComponentVisualizer):
    """Finds and visualizes concepts based on activation maximization.

    This class implements the activation-based approach to component
    visualization. It processes a dataset to find the input examples that
    produce the highest activation values for each component within specified
    layers of a neural network.

    The results, including the indices of the top-activating samples, are
    cached to disk for efficient re-use in subsequent analyses.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to analyze. It is recommended that the model
        has a `.name` attribute for reliable caching.
    dataset_model : torch.utils.data.Dataset
        The dataset used for model inference to find top-activating samples.
        It should be preprocessed as required by the model. It is recommended
        that the dataset has a `.name` attribute for reliable caching.
    dataset_fm : torch.utils.data.Dataset
        The dataset preprocessed for the foundation model. This version should
        yield raw data (e.g., PIL Images) that the foundation model's own
        preprocessor can handle.
    layer_names : list[str]
        A list of names of the layers to analyze (e.g., `['layer4.1.conv2']`).
    num_samples : int
        The number of top-activating samples to collect for each component.
    device : torch.device or str, optional
        The device on which to perform computations. If None, the model's
        current device is used.
    aggregate_fn : callable, optional
        A function to aggregate the spatial or temporal dimensions of the layer
        activations into a single value per component. If None, defaults to
        taking the mean over spatial dimensions for convolutional layers.
        (A selection of aggregation functions are provided in
        `semanticlens.component_visualization.aggregators`.)
    cache_dir : str or None, optional
        The root directory for caching results. If None, caching is disabled.

    Attributes
    ----------
    actmax_cache : ActMaxCache
        An object that manages the collection and caching of top activations.

    Raises
    ------
    ValueError
        If any layer name in `layer_names` is not found in the model.

    Examples
    --------
    >>> import torch
    >>> from torchvision.models import resnet18
    >>> from torch.utils.data import TensorDataset
    >>> from semanticlens.component_visualization import ActivationComponentVisualizer
    >>>
    >>> # 1. Prepare model and dataset
    >>> model = resnet18(weights=...)
    >>> model.name = "resnet18"
    >>> dummy_data = TensorDataset(torch.randn(100, 3, 224, 224))
    >>> dummy_data.name = "dummy_data"
    >>>
    >>> # 2. Initialize the visualizer
    >>> visualizer = ActivationComponentVisualizer(
    ...     model=model,
    ...     dataset_model=dummy_data,
    ...     dataset_fm=dummy_data, # Using same dataset for simplicity here
    ...     layer_names=["layer4.1.conv2"],
    ...     num_samples=10,
    ...     cache_dir="./cache"
    ... )
    >>>
    >>> # 3. Run the analysis to find top-activating samples
    >>> # This will process the dataset and save the results to the cache.
    >>> # visualizer.run(batch_size=32)
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
        self._init_cache_dir(cache_dir)
        self._validate_args()

        self.layer_names = layer_names
        self._check_layers()

        device = device or next(model.parameters()).device
        self.model.to(device)

        if aggregate_fn is None:
            logger.warning(f"No aggregation_fn provided using default: {aggregators.aggregate_conv_mean.__name__}")
            aggregate_fn = aggregators.aggregate_conv_mean

        self.actmax_cache = ActMaxCache(self.layer_names, n_collect=num_samples, aggregation_fn=aggregate_fn)

        if self.caching:
            try:
                self.actmax_cache.load(self.storage_dir)
                logger.info(f"Results loaded from {self.storage_dir}")
            except FileNotFoundError:
                logger.info(f"Results will be stored in {self.storage_dir}")

    def _validate_args(self):
        """For caching we need names for model and dataset.
        They are supposed to be provided as instance-attributes.
        If they are missing we use a fallback that is a combination of their class-name and a hash of their printable representation (`repr()`).
        """
        if not hasattr(self.model, "name"):
            model_name = get_fallback_name(self.model)
            if self.caching:
                message = (
                    f"Model does not have a name attribute, which is required for reliable caching.\n"
                    f"Using a fallback name: {model_name}."
                )
                warnings.warn(message, MissingNameWarning, stacklevel=2)
            self.model.name = model_name
        if not hasattr(self.dataset, "name"):
            dataset_name = get_fallback_name(self.dataset)
            if self.caching:
                message = (
                    f"Dataset does not have a name attribute, which is required for reliable caching.\n"
                    f"Using a fallback name: {dataset_name}."
                )
                warnings.warn(message, MissingNameWarning, stacklevel=2)
            self.dataset.name = dataset_name

        if len(self.dataset) != len(self.dataset_fm):
            raise ValueError(
                "Model and foundation model datasets should have the same length.",
                (len(self.dataset), len(self.dataset_fm)),
            )

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
        """Run the activation maximization analysis on the dataset.

        This method processes the entire `dataset_model` to find the maximally
        activating input examples for each component in the specified layers.
        If a valid cache is found, the results are loaded directly from disk,
        skipping the computation. Otherwise, the computation is performed and
        the results are saved to the cache.

        Parameters
        ----------
        batch_size : int, default=32
            The batch size to use for processing the dataset.
        num_workers : int, default=0
            The number of worker processes for the data loader.

        Returns
        -------
        dict
            A dictionary mapping layer names to `ActMax` instances, which
            contain the top activating samples for each component.
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
    def _run(self, batch_size: int = 64, num_workers: int = 0):
        """Actuall ActMax-Cache computation/population and caching."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        with self.actmax_cache.hook_context(self.model):
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Collecting ActMax"):
                _ = self.model(images.to(self.device)).cpu()

        if self._cache_root:
            self.actmax_cache.store(self.storage_dir)
            logger.debug(f"Stored activation maximization cache at {self.storage_dir}")

        return self.actmax_cache.cache

    @torch.no_grad()
    def _compute_concept_db(self, fm, batch_size=32, **kwargs):
        """
        Compute the concept database for the given fm (foundation model).

        This is called from the Lens class following the Inversion of Control pattern.
        The method processes the dataset to find maximally activating samples and then
        embeds those samples using the foundation model.

        Parameters
        ----------
        fm : FoundationModel
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
        self.run(batch_size=batch_size, **kwargs)

        embeds = self._embed_vision_dataset(fm, batch_size, **kwargs)

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

        """
        fm.to(self.device)

        def pil_list_collate(batch):
            """We apply the FM transformation (via fm.preprocess) lazy, thus the dataset_fm returns PILs and a special collate implemenation is needed."""
            if isinstance(batch[0], (tuple, list)):
                return [item[0] for item in batch]
            return list(batch)

        pil_dataloader = torch.utils.data.DataLoader(
            self.dataset_fm, batch_size=batch_size, shuffle=False, collate_fn=pil_list_collate, **kwargs
        )
        embeds = []
        with tqdm(total=len(self.dataset), desc="Embedding Dataset") as pbar_dataset:
            for pil_list in pil_dataloader:
                inputs = fm.preprocess(pil_list)
                fm_out = fm.encode_image(inputs).cpu()
                embeds.append(fm_out)
                pbar_dataset.update(batch_size)
        embeds = torch.cat(embeds)

        assert embeds.shape[0] == len(self.dataset_fm), "Number of embeddings does not match number of ids!"
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

    def visualize_components(
        self,
        component_ids: torch.Tensor,
        layer_name: str,
        n_samples: int = 9,
        nrows: int = 3,
        fname=None,
        denormalization_fn=None,
    ):
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
        denormalization_fn : callable, optional
            Function to denormalize the images before visualization.
        """
        self._check_layer_name(layer_name)
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        if hasattr(self.dataset, "denormalization_fn"):
            post_process = self.dataset.denormalization_fn
        elif denormalization_fn is not None:
            post_process = denormalization_fn
        else:
            logger.debug("Dataset does not have denormalization_fn method.")

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

        plt.suptitle((f"{fname:.15} " if fname else "") + f"{self.model.name:>.10} {layer_name:<.15}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        if self.caching:
            component_id_str = "-".join(map(str, component_ids.tolist()))
            fdir = self.storage_dir / "plots"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / ((fname + "_" if fname else "") + f"{layer_name}_{component_id_str}.png")
            plt.savefig(fpath)
            plt.close(fig)
            print(f"Saved visualization to {fpath}")
        elif fname:
            logger.warning(
                f"Failed to save visualization to {fpath} caching is not enabled in the ComponentVisualizer (`cv.caching: False`)"
            )

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
