"""
Relevance-based component visualization using attribution methods.

This module provides tools for visualizing neural network components using
Layer-wise Relevance Propagation (LRP) and Concept Relevance Propagation (CRP)
attribution methods to understand which input features are most relevant for
specific neural activations.
"""

from __future__ import annotations

import logging
import warnings

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from crp.cache import Cache
from crp.concepts import ChannelConcept as Concept
from crp.helper import load_maximization
from crp.maximization import Maximization
from crp.statistics import Statistics
from crp.visualization import FeatureVisualization
from crp.image import plot_grid


from zennit.composites import EpsilonPlusFlat as Composite

from semanticlens.component_visualization import aggregators
from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.render import crop_and_mask_images, binary_mask_for_crop_and_mask_images
from semanticlens.utils.helper import get_fallback_name

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class MissingNameWarning(UserWarning):
    """
    Warning raised when a model or dataset is missing a `.name` attribute.

    This attribute is crucial for the caching mechanism to create a stable and predictable cache location.
    Without it, a fallback name is generated.
    """

    pass

class RelevanceComponentVisualizer(AbstractComponentVisualizer):
    """
    Find concepts based on activation maximization and further process the concepts based on relevance-based analysis.

    This class implements the relevance-based approach to component visualization.
    It processes a dataset to find the input examples that produce 
    the highest activation values for each component within specified
    layers of a neural network. It further crops the identified input examples
    based on relevance attribution maps.

    Parameters
    ----------
    attribution : crp.attribution.Attributor
        Attribution method for computing relevance scores.
    dataset_model : torch.utils.data.Dataset
        Dataset for top-activating samples, preprocessed as required by the model.
        It is recommended that the dataset has a `.name` attribute for reliable caching.
    dataset_fm : torch.utils.data.Dataset
        Dataset preprocessed for foundation model. Should yield raw data
        (e.g., PIL Images) that the foundation model's preprocessor can handle.
    layer_names : list[str]
        List of names of the layers to analyze (e.g., `['layer4']`).
    num_samples : int
        Number of top-activating samples to collect for each component.
    preprocess_fn : callable
        Function for preprocessing dataset for top-activating samples. (arg for FeatureVisualization)
    composite : zennit.composites.Composite, optional
        Composite rule for attribution computation.
    aggregate_fn : str, optional
        Function for aggregating activations. If None, uses default conv mean aggregation.
    abs_norm : bool, default=True
        Whether to use absolute normalization during relevance computation.
    device : torch.device or str, optional
        Device for computations. If None, uses model's current device.
    cache_dir : str, optional
        Directory for caching results. Default is "./cache".
    cache : optional
        Additional caching configuration for the underlying CRP framework.
    plot_fn : callable, default=crop_and_mask_images
        Function for plotting visualizations.

    Attributes
    ----------
    agg_fn_name : str
        Name of the aggregation function.
    plot_fn_name : str
        Name of the plotting function.
    feature_visualization : 
        Instance of the feature visualization class from zennit-crp.
        
    Methods
    -------
    run(data_start=0, data_end=None, batch_size=32, checkpoint=500, on_device=None)
        Identify maximally activating examples and run relevance-based preprocessing.
    _compute_concept_db(fm, component_ids=None, batch_size=32)
        Compute the concept (component) database for the given fm (foundation model).
    get_max_reference(layer_name)
        Get sample IDs of maximally activating samples (i.e. reference samples) for a layer.
    get_masks_for_max_reference(component_ids, layer_name, batch_size=32)
        Get masks for each reference sample based on relevance analysis.
    get_max_reference_examples(component_ids, layer_name, batch_size=32)
        Get reference samples processed based on relevance analysis.  (e.g cropping)

    _init_cache_dir(cache_dir)
        Initialize the cache directory for storing results.
    _get_function_name(func, purpose)
        Get the name of the function.
    _check_layers()
        Validate that all specified layers exist in the model.
    _check_layer_name(layer_name):
        Validate that a layer name exists in the configured layers.

    Properties
    ----------
    metadata : dict
        Metadata about the visualizer configuration.
    storage_dir: pathlib.Path
        Directory for storing feature visualization results.

    Examples
    --------
    >>> import torch
    >>> from crp.attribution import CondAttribution
    >>> from zennit.canonizers import SequentialMergeBatchNorm
    >>> from zennit.composites import EpsilonPlusFlat
    >>> from semanticlens.component_visualization import RelevanceComponentVisualizer
    >>>
    >>> # 1. Prepare attribution method and datasets
    >>> attribution = CondAttribution(model)
    >>> dataset_model.name = "imagenet"
    >>> composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    >>>
    >>> # 2. Initialize the visualizer
    >>> visualizer = RelevanceComponentVisualizer(
    ...     attribution=attribution,
    ...     dataset_model=dataset_model,
    ...     dataset_fm=dataset_fm,
    ...     layer_names=["layer4"],
    ...     num_samples=10,
    ...     preprocess_fn=preprocess_fn,
    ...     composite=composite,
    ... )
    >>>
    >>> # 3. Run the analysis
    >>> visualizer.run()
    """

    def __init__(
        self,
        attribution,
        dataset_model,
        dataset_fm,
        layer_names: list[str],
        num_samples: int,
        preprocess_fn: Callable,
        composite: Optional[Composite] = None,
        aggregate_fn: str = None,
        abs_norm: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        cache_dir: str = "./cache",
        cache: Optional[Cache] = None,
        plot_fn: Callable = crop_and_mask_images,
    ):
        """
        Initialize the relevance component visualizer.

        Parameters
        ----------
        attribution : crp.attribution.Attributor
            Attribution method for computing relevance scores.
        dataset_model : torch.utils.data.Dataset
            Dataset for top-activating samples.
        dataset_fm : torch.utils.data.Dataset
            Dataset preprocessed for foundation model.
        layer_names : list[str]
            Names of the layers to analyze.
        num_samples : int
            Number of top-activating samples per component.
        preprocess_fn : callable
            Function for preprocessing dataset for top-activating samples. (arg for FeatureVisualization)
        composite : zennit.composites.Composite, optional
            Composite rule for attribution computation.
        aggregate_fn : str, optional
            Function for aggregating activations.
        abs_norm : bool, default=True
            Whether to use absolute normalization during relevance computation.
        device : torch.device or str, optional
            Device for computations.
        cache_dir : str, optional
             Directory for caching results.
        cache : Cache, optional
            Additional caching configuration for relevance-based visualization.
        plot_fn : callable, default=crop_and_mask_images
            Function for plotting visualizations.
        """
        self.model = attribution.model
        self.dataset = dataset_model
        self.dataset_fm = dataset_fm
        self._init_cache_dir(cache_dir)
        self._validate_args()

        self.layer_names = layer_names
        self._check_layers()

        device = device or next(self.model.parameters()).device
        self.model.to(device)

        if aggregate_fn is None:
            logger.warning(f"No aggregation_fn provided using default: sum")
            aggregate_fn = 'sum'
        
        self.agg_fn_name = aggregate_fn
        self.plot_fn_name = self._get_function_name(plot_fn, "Plot function")

        self.num_samples = num_samples
        self.composite = composite
        self.plot_fn = plot_fn
        self.abs_norm = abs_norm


        # Create CRP objects using composition (not inheritance)
        self.feature_visualization = FeatureVisualization(
            attribution=attribution,
            dataset=dataset_model,
            layer_map={layer_name: Concept() for layer_name in self.layer_names},
            preprocess_fn=preprocess_fn,
            max_target=aggregate_fn,
            abs_norm=abs_norm,
            path=self.storage_dir,
            device=device,
            cache=cache,
        )

        # overwrite ActMax and ActStats: set abs_norm = False and SAMPLE_SIZE = num_samples for activation maximization
        setattr(self.feature_visualization, "ActMax", \
                Maximization(mode="activation", max_target=aggregate_fn, abs_norm=False, path=self.storage_dir))
        setattr(self.feature_visualization, "ActStats", \
                Statistics(mode="activation", max_target=aggregate_fn, abs_norm=False, path=self.storage_dir))
        setattr(self.feature_visualization.ActMax, "SAMPLE_SIZE", self.num_samples)
        setattr(self.feature_visualization.ActStats, "SAMPLE_SIZE", self.num_samples)

        if self.caching:
            logger.info(f"Preprocessed data already available in {self.storage_dir}")
        else:
            logger.info(f"Preprocessed data will be stored in {self.storage_dir}")

    def _validate_args(self):
        """For caching we need names for model and dataset.
        They are supposed to be provided as instance-attributes.
        If they are missing we use a fallback that is a combination of their class-name and a hash of their printable representation (`repr()`).
        """
        if not hasattr(self.model, "name"):
            model_name = get_fallback_name(self.model)
            message = (
                f"Model does not have a name attribute, which is required for reliable caching.\n"
                f"Using a fallback name: {model_name}."
            )
            warnings.warn(message, MissingNameWarning, stacklevel=2)
            self.model.name = model_name
        if not hasattr(self.dataset, "name"):
            dataset_name = get_fallback_name(self.dataset)
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

    def run(
        self,
        data_start: int = 0,
        data_end: Optional[int] = None,
        batch_size: int = 32,
        checkpoint: int = 500,
        on_device: Optional[Union[torch.device, str]] = None,
    ) -> dict[str, list[str]]:
        """
        Identify maximally activating examples and run relevance-based preprocessing.

        Identify maximally activating examples and processes the dataset 
        using attribution methods to compute relevance scores for each component.

        Parameters
        ----------
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

        Returns
        -------
        list or other
            Results from preprocessing, or list of existing files if already preprocessed.

        Raises
        ------
        AssertionError
            If no composite rule is provided.
        """

        if not self.caching:
            logger.info("Preprocessing...")
            data_end = len(self.feature_visualization.dataset) if data_end is None else data_end

            # Delegate to FeatureVisualization object
            return self.feature_visualization.run(self.composite, data_start, data_end, batch_size, checkpoint, on_device)
        else:
            logger.info("Already preprocessed")
            return self._reconstruct_saved_checkpoints()

    def _reconstruct_saved_checkpoints(self) -> dict[str, list[str]]:
        """
        Reconstruct saved_checkpoints structure from cached files.

        Returns
        -------
        dict[str, list[str]]
            Dictionary containing file path lists for each processing type
        """
        saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        
        # Storage paths for each component type
        paths = {
            "r_max": self.feature_visualization.RelMax.PATH,
            "a_max": self.feature_visualization.ActMax.PATH,
            "r_stats": self.feature_visualization.RelStats.PATH,
            "a_stats": self.feature_visualization.ActStats.PATH
        }
        
        # Check if at least one cache directory exists
        if not any(os.path.exists(path) for path in paths.values()):
            raise FileNotFoundError(f"No cache directories found. Expected paths: {list(paths.values())}")

        for key, path in paths.items():
            if os.path.exists(path):
                if "stats" in key:
                    for layer_name in self.layer_names:
                        layer_path = os.path.join(path, layer_name)
                        all_files = os.listdir(layer_path)
                        saved_checkpoints[key] = \
                            [os.path.join(layer_path, f.replace("data.npy", "")) for f in all_files if f.endswith("data.npy")]
                else:
                    all_files = os.listdir(path)
                    saved_checkpoints[key] = \
                        [os.path.join(path, f.replace("data.npy", "")) for f in all_files if f.endswith("data.npy")]

        return saved_checkpoints


    def _compute_concept_db(self, fm, component_ids: Optional[dict[str, Union[list[int], int]]] = None, batch_size=32, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the concept (component) database for the given fm (foundation model).

        This is called from the Lens class following the Inversion of Control pattern.
        The method processes the dataset to find maximally activating samples and then
        embeds those samples using the foundation model.

        Parameters
        ----------
        fm : FoundationModel
            The foundation model used for embedding the maximally activating samples.
        component_ids : dict[str, Union[list[int], int]], optional
            Dictionary mapping layer names to lists of component IDs to include in the component database. If None, all components are included.
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
        
        if component_ids is None:
            component_ids = {}
            for layer_name in self.layer_names:
                num_components = self.get_max_reference(layer_name).shape[0]
                component_ids[layer_name] = list(range(num_components))

        concept_db = dict()
        for layer_name in self.layer_names:
            component_ids_this_layer = component_ids[layer_name]
            if isinstance(component_ids_this_layer, int):
                component_ids_this_layer = [component_ids_this_layer]
            reference_ids = self.get_max_reference(layer_name)[component_ids_this_layer, :]
            reference_masks = self.get_masks_for_max_reference(component_ids_this_layer, layer_name, batch_size)
            embeds = self._embed_vision_dataset(fm, reference_ids, reference_masks, **kwargs)
            concept_db[layer_name] = embeds
        return concept_db

    def _embed_vision_dataset(self, fm, reference_ids: torch.Tensor, reference_masks: dict[int, list[Image.Image]], **kwargs) -> torch.Tensor:
        """
        Embed the vision dataset using the provided foundation model.

        Parameters
        ----------
        fm : FoundationModel
            Foundation model with encode_image method for embedding images.
        reference_ids: torch.Tensor
            Tensor of shape (n_components, n_samples) containing reference ids for each component.
        reference_masks: dict[int, list[Image.Image]]
            Dictionary mapping component IDs to their masks.
        **kwargs
            Additional keyword arguments passed to DataLoader.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components, n_samples, embedding_dim) containing embeddings
            for reference samples for each component.
        """

        fm.to(self.device)

        assert reference_ids.shape[0] == len(reference_masks), \
            "The # of components does not match between reference_ids and reference_masks"
        
        all_component_embeds = []
        for loc, component_id in enumerate(reference_masks.keys()):
            reference_ids_this_component = reference_ids[loc, :].tolist()
            reference_masks_this_component = reference_masks[component_id]
            dataloader = self._vision_dataset_for_embed(
                reference_ids_this_component,
                reference_masks_this_component,
                **kwargs
            )
            # Collect embedings for this component
            component_embeds = []
            for pil_list in dataloader:
                inputs = fm.preprocess(pil_list)
                fm_out = fm.encode_image(inputs).cpu()
                component_embeds.append(fm_out)
            
            # Concatenate all batches for this component
            component_tensor = torch.cat(component_embeds, dim=0)  # Shape: (n_samples, embedding_dim)
            all_component_embeds.append(component_tensor)        # Stack all components to create final tensor

        embeds = torch.stack(all_component_embeds, dim=0)  # Shape: (n_components, n_samples, embedding_dim)

        return embeds

    def _vision_dataset_for_embed(self,
                                  reference_ids: list[int],
                                  masks: list[Image.Image], 
                                  batch_size: int = 32, 
                                  **kwargs):
        """
        Create a DataLoader for the vision dataset.

        Parameters
        ----------
        reference_ids : list[int]
            Reference image ids.
        masks : list[Image.Image]
            List of masks for each reference image.
        batch_size : int, default=32
            Batch size for processing.
        **kwargs
            Additional keyword arguments passed to DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for the vision dataset.
        """

        def pil_list_collate(batch):
            """We apply the FM transformation (via fm.preprocess) lazy, thus the dataset_fm returns PILs and a special collate implementation is needed."""
            if isinstance(batch[0], (tuple, list)):
                return [item[0] for item in batch]
            return list(batch)
        
        # Create a cropped dataset
        cropped_dataset = CroppedDataset(
            base_dataset=self.dataset_fm,
            reference_ids=reference_ids,
            masks=masks,
        )
        
        # Create DataLoader for the pre-cropped dataset 
        return torch.utils.data.DataLoader(
            cropped_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=pil_list_collate, 
            **kwargs)   

    def get_max_reference(self, layer_name) -> torch.Tensor:
        """
        Get sample IDs of maximally activating samples (reference samples) for a layer.

        This uses the results from self.run (more precisely FeatureVisualization.run from zennit-crp).
        
        This method takes the same input/output as ActivationComponentVisualizer.get_max_reference in the activation_based.py,
        which is DISTINCT from the FeatureVisualization.get_max_reference in zennit-crp.

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
        # load maximally activating reference ids
        d_c_sorted, _, _ = load_maximization(self.feature_visualization.ActMax.PATH, layer_name)
        return d_c_sorted[0:self.num_samples, :].T # only the first num_samples references

    def get_masks_for_max_reference(self, component_ids: int | list, layer_name: str, batch_size: int = 32) -> dict[int, list[Image.Image]]:
        """
        Get masks for each reference sample based on relevance analysis.

        Computes relevance-based visualizations for specified components using
        attribution methods to highlight the most relevant input features.

        This uses a custom masking function for get_max_reference in zennit-crp.

        Parameters
        ----------
        component_ids : int or list of int
            IDs of components to visualize.
        layer_name : str
            Name of the layer containing the components.
        n_ref : int
            Number of reference examples to generate.
        batch_size : int, default=32
            Batch size for processing.

        Returns
        -------
        dict
            Dictionary mapping component IDs to their reference example masks.

        Raises
        ------
        ValueError
            If mask functions for the specified plot function are not supported.
        AttributeError
            If gradients are not enabled or CRP requirements are not met.

        Notes
        -----
        This method requires gradients to be enabled for LRP/CRP computation.
        The `torch.enable_grad()` decorator ensures this requirement is met.
        """
        mode = "activation"
        r_range = (0, self.num_samples)
        composite = self.composite
        rf = True

        if getattr(self.plot_fn, '__name__', None) == 'crop_and_mask_images':
            mask_fn = binary_mask_for_crop_and_mask_images
        else: 
            raise ValueError(f"Binary mask function for {self.plot_fn} is not defined. Currently only 'crop_and_mask_images' is supported.")

        try:
            return self.feature_visualization.get_max_reference(component_ids, layer_name, mode, r_range, composite, rf, mask_fn, batch_size)
        except AttributeError as e:
            logging.error("Error during LRP/CRP-based concept-visualization.")
            logging.error("Note `crp` requires gradients: Make sure to execute with torch autograd enabled.")
            raise e

    def get_max_reference_examples(self, component_ids: int | list, layer_name: str, batch_size: int = 32) -> dict[int, list[Image.Image]]:
        """
        Get reference samples processed based on relevance analysis.  (e.g cropping)
        
        Computes relevance-based visualizations for specified components using
        attribution methods to highlight the most relevant input features.

        This utilises get_max_reference in zennit-crp.

        Parameters
        ----------
        component_ids : int or list of int
            IDs of components to visualize.
        layer_name : str
            Name of the layer containing the components.
        n_ref : int
            Number of reference examples to generate.
        batch_size : int, default=32
            Batch size for processing.

        Returns
        -------
        dict
            Dictionary mapping component IDs to their reference example masks.

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
        r_range = (0, self.num_samples)
        composite = self.composite
        rf = True
        self._check_layer_name(layer_name)

        try:
            return self.feature_visualization.get_max_reference(component_ids, layer_name, mode, r_range, composite, rf, self.plot_fn, batch_size)
        except AttributeError as e:
            logging.error("Error during LRP/CRP-based concept-visualization.")
            logging.error("Note `crp` requires gradients: Make sure to execute with torch autograd enabled.")
            raise e

    def visualize_components(
        self,
        component_ids: torch.Tensor,
        layer_name: str,
        n_samples: int = 9,
        #nrows: int = 3,
        fname: str =None,
        #denormalization_fn=None,
        **kwargs,
    ):
        """
        Visualize specific components by displaying their top activating samples using plot_grid from crp.image.

        A good place to put it here since we need access to the PIL-dataset to implement this. However we should call a stateless function in here that abstracts complexity and can be used by other versions of the concept visualizer as well.

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
        fname : str, optional
            Filename to save the visualization. If None, the plot is not saved.
        denormalization_fn : callable, optional
            Function to denormalize the images before visualization.
        **kwargs
            Additional keyword arguments passed to plot_grid.
        """
        
        #TODO: add denormalization_fn (and nrows?) support

        if n_samples > self.num_samples:
            raise ValueError(f"n_samples ({n_samples}) cannot be larger than the number of collected samples ({self.num_samples}).")

        ref_c = self.get_max_reference_examples(component_ids.to(int).tolist(), layer_name)

        # extract only the first n_samples for each component
        ref_c_samples = {}
        for c_id in ref_c.keys():
            ref_c_samples[c_id] = ref_c[c_id][0:n_samples]

        plot_grid(ref_c_samples, **kwargs)
        
        if fname:
            fig = plt.gcf()

            component_id_str = "-".join(map(str, component_ids.tolist()))
            fdir = self.storage_dir / "plots"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / ((fname + "_" if fname else "") + f"{layer_name}_{component_id_str}.png")
            
            fig.savefig(fpath)
            plt.close(fig)
            print(f"Saved visualization to {fpath}")


    def to(self, device: Union[torch.device, str]):
        """
        Move visualizer and attribution model to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device for the visualizer and attribution model.

        Returns
        -------
        RelevanceComponentVisualizer
            Returns self for method chaining.
        """
        self.device = device
        self.model.to(device)
        self.feature_visualization.attribution.model.to(device)
        return self

    @property
    def metadata(self) -> dict[str, str]:
        """
        Get metadata about the visualizer configuration.

        This property provides a dictionary containing all configuration
        parameters necessary for caching and reproducibility.

        Returns
        -------
        dict[str, str]
            Dictionary containing metadata about the cache, dataset, and model.
        """
        return {
            "aggregate_fn_name": str(self.agg_fn_name),
            "n_collect": str(self.num_samples),
            "layer_names": str(self.layer_names),
            "dataset": str(self.dataset.name),
            "model": str(self.model.name),
            "plot_fn": str(self.plot_fn_name),
            "abs_norm": str(self.abs_norm),
        }

    @property
    def caching(self) -> bool:
        """Check if cached data available or not."""
        # Storage paths for each component type
        paths = [
            self.feature_visualization.RelMax.PATH,
            self.feature_visualization.ActMax.PATH,
            self.feature_visualization.RelStats.PATH,
            self.feature_visualization.ActStats.PATH
        ]

        # Check if all directories exist
        path_flag = bool(all(os.path.exists(path) for path in paths))

        # check if files exist
        _flags = []
        for layer_name in self.layer_names:
            for path in paths:
                if os.path.exists(path):
                    all_files = os.listdir(path)
                    _flags.append(any([f.startswith(layer_name) for f in all_files]))

        file_flag = all(_flags)
        
        return path_flag and file_flag
    
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
        return self._cache_root / self.__class__.__name__ / self.dataset.name / self.model.name

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

    def _init_cache_dir(self, cache_dir: str):
        """
        Initialize the cache directory for storing results.

        Parameters
        ----------
        cache_dir : str
            Directory path for caching.
        """
        self._cache_root = Path(cache_dir)
        self._cache_root.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _get_function_name(func: Callable, purpose: str) -> str:
        """
        Get the name of the function.

        Parameters
        ----------
        func : callable
            The function to get the name of.
        purpose: str
            A description of what the function is for (e.g., "plot function").

        Returns
        -------
        str
            The name of the function.

        Raises
        ------
        ValueError
            If the function is a lambda function.
        """
        func_name = getattr(func, "__name__", None)
        if func_name is None or func_name == "<lambda>":
            raise ValueError(f"{purpose} must be a defined function, not a lambda.")
        return func_name

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

    def _check_layer_name(self, layer_name: str):
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


class CroppedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that pre-applies cropping based on masks.

    This dataset takes a base dataset, subsets it with reference_ids, and applies
    cropping to all samples during initialization for better performance.
    """

    def __init__(self, base_dataset: torch.utils.data.Dataset, reference_ids: list[int], masks: list[Image.Image]):
        """
        Initialize the cropped dataset.

        Parameters
        ----------
        base_dataset : torch.utils.data.Dataset
            The base dataset containing PIL images. (Width, Height)
        reference_ids : list[int]
            List of reference indices from the base dataset.
        masks : list[Image.Image]
            List of masks with shape (Width, Height).

        Note
        ----
        Assuming the base_dataset is provided to foundation models and ToTensor() has not been applied yet, the images will be in (Width, Height) format.
        Masks obtained from relevance-based analysis should also be in (Width, Height) format.
        """
        self.base_dataset = base_dataset
        self.reference_ids = reference_ids
        self.masks = masks

        assert len(reference_ids) == len(masks), \
            f"Number of reference IDs ({len(reference_ids)}) must match number of masks ({len(masks)})"
        
        # Pre-compute all cropped images during initialization
        self.cropped_images = self._precompute_cropped_images()
    
    def _precompute_cropped_images(self) -> list:
        """
        Pre-compute all cropped images and store them in memory.
        
        Returns
        -------
        list
            List of pre-cropped PIL images or (image, label) tuples if labels are present.
        """
        cropped_images = []

        for idx in range(len(self.reference_ids)):
            # Get the original sample from base dataset
            original_reference_id = self.reference_ids[idx]
            original_reference = self.base_dataset[original_reference_id]
            
            # Handle case where dataset returns (image, label) tuple
            if isinstance(original_reference, (tuple, list)):
                img = original_reference[0]
                label = original_reference[1]
            else:
                img = original_reference
                label = None

            # Get the corresponding mask for this sample
            mask = self.masks[idx]

            assert mask.size == img.size, "The mask size must match the image size."

            # Apply cropping
            cropped_image = self.crop_pil_with_mask(img, mask)
    
            if label is not None:
                cropped_images.append((cropped_image, label))
            else:
                cropped_images.append(cropped_image)
        
        return cropped_images
    
    @staticmethod
    def crop_pil_with_mask(pil_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Crop PIL image using binary PIL mask.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            The input PIL image to crop.
        mask : PIL.Image.Image
            The binary mask PIL image.

        Returns
        -------
        PIL.Image.Image
            The cropped PIL image.

        Raises
        ------
        ValueError
            If the mask is not a filled rectangle.
        """
        # check if the mask is a filled rectangle or not.
        mask_np = np.array(mask).astype(bool)
        
        # find bounding box
        true_indices = np.argwhere(mask_np)
        min_row, min_col = true_indices.min(axis=0) 
        max_row, max_col = true_indices.max(axis=0)

        # Check if all values are True inside the box
        if not np.all(mask_np[min_row:max_row+1, min_col:max_col+1]):
            raise ValueError("Mask is not a filled rectangle.")
    
        # Apply mask to pil_image
        pil_image_np = np.array(pil_image)
        cropped_region = pil_image_np[min_row:max_row+1, min_col:max_col+1].copy()

        return Image.fromarray(cropped_region)
    
    def __len__(self):
        return len(self.cropped_images)

    def __getitem__(self, idx: int):
        """
        Get a pre-cropped image by index.
        
        Parameters
        ----------
        idx : int
            Index of the image to retrieve.
            
        Returns
        -------
        PIL.Image
            Pre-cropped PIL image.
        """
        return self.cropped_images[idx]