"""
Lens: Main class for visual concept analysis and exploration.

This module provides the primary interface for semantic analysis of neural networks,
combining component visualization with foundation models to explore relationships
between visual concepts and text embeddings.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path

import einops
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import semanticlens as sl
from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.foundation_models.base import VisionLanguageFoundationModel

logger = logging.getLogger(__name__)

# TODO Device

# helpers ---


def consistent_hash(value):
    """
    Generate a consistent SHA256 hash for a string value.

    Parameters
    ----------
    value : str
        String value to hash.

    Returns
    -------
    str
        Hexadecimal hash digest.
    """
    return hashlib.sha256(value.encode()).hexdigest()


def cache_image_dataset_embeddings(method):
    """
    Decorator for caching image dataset embeddings.

    This decorator wraps methods that compute image dataset embeddings to provide
    automatic caching functionality. If cached results exist and match the current
    configuration, they are loaded instead of recomputing.

    Parameters
    ----------
    method : callable
        Method to wrap with caching functionality.

    Returns
    -------
    callable
        Wrapped method with caching capability.
    """

    @functools.wraps(method)
    def wrapper(self, *args, cache=True, **kwargs):
        if cache:
            _, data_ids = self._get_dataset_and_ids()
            fpath = Path(self.storage_dir) / "image_embeds" / self.fm.name / f"{self.dataset_name}.safetensors"
            if fpath.exists():
                try:
                    cached = sl.ConceptTensor.load(fpath)
                    if (
                        cached.metadata["foundation_model"] == self.fm.name
                        and cached.metadata["dataset"] == self.dataset_name
                        and cached.metadata["ids"] == data_ids
                    ):
                        logger.debug("Loaded from cache.")
                        return cached
                    else:
                        logger.debug("Cache mismatch, recomputing!")
                except Exception as e:
                    logger.error("Caching mechanism failed: %s", e)
        result = method(self, *args, cache=False, **kwargs)

        if cache:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            result.save(fpath)
            logger.debug("Saved to cache at: %s", fpath)

        return result

    return wrapper


def cache_component_specific_embeddings(method):
    """
    Decorator for caching component-specific embeddings.

    This decorator wraps methods that compute component-specific embeddings to
    provide automatic caching functionality based on component visualizer metadata
    and model configuration.

    Parameters
    ----------
    method : callable
        Method to wrap with caching functionality.

    Returns
    -------
    callable
        Wrapped method with caching capability.
    """

    @functools.wraps(method)
    def wrapper(self, batch_size, composite, n_ref, rf, layer_name, cache):
        if cache:
            fpath = (
                Path(self.storage_dir)
                / "component_specific_embeds"
                / self.dataset_name
                / f"{self.fm.name}_{layer_name}.safetensors"
            )
            if fpath.exists():
                try:
                    cached = sl.ConceptTensor.load(fpath)
                    if (
                        cached.metadata["component_visualizer"] == json.dumps(self.cv.metadata, sort_keys=True)
                        and cached.metadata["foundation_model"] == self.fm.name
                        and cached.metadata["n_reference_samples"] >= (n_ref or 0)
                    ):
                        logger.debug("Loaded from cache.")
                        return cached[:, :n_ref]
                    else:
                        logger.debug("Cache mismatch, recomputing!")
                except Exception as e:
                    logger.error("Caching mechanism failed:", e)

        result = method(self, batch_size, composite, n_ref, rf, layer_name, cache=False)

        if cache:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            result.save(fpath)
            logger.debug("Saved to cache at: %s", fpath)

        return result

    return wrapper


# main ---


class Lens:
    """
    Main class for visual concept analysis and exploration.

    Lens provides methods to embed images and text datasets for semantic analysis,
    integrating component visualization with foundation models to explore relationships
    between visual concepts and text embeddings. It serves as the primary interface
    for semantic analysis of neural networks.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset containing input examples for analysis.
    component_visualizer : AbstractComponentVisualizer
        Visualizer for analyzing neural network components.
    foundation_model : VisionLanguageFoundationModel
        Foundation model for encoding images and text.
    dataset_name : str
        Name identifier for the dataset.
    storage_dir : str or Path
        Directory for caching computed embeddings and results.
    device : torch.device or str, optional
        Device for computation. If None, uses the foundation model's device.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Input dataset for analysis.
    cv : AbstractComponentVisualizer
        Component visualizer instance.
    fm : VisionLanguageFoundationModel
        Foundation model instance.
    fm_preprocessor : VisionLanguageProcessor
        Preprocessor for the foundation model.
    layer_names : list of str
        Names of layers being analyzed.
    dataset_name : str
        Dataset identifier.
    device : torch.device
        Computation device.
    storage_dir : str or Path
        Storage directory for caching.

    Properties
    ----------
    concept_db : dict
        Database of computed concept embeddings by layer.

    Methods
    -------
    embed_text_dataset(texts, templates=["an image of {}"], batch_size=32, cache=True, device=None)
        Embed a list of text concepts, applying templates if provided.
    embed_text(text, templates=None, device=None)
        Embed a single text prompt.
    embed_image_dataset(batch_size=32, device=None, cache=True, **dataloader_kwargs)
        Embed an image dataset using the foundation model.
    compute_semantic_embeddigs(layer_names, component_specific_examples=False, batch_size=32, device=None, cache=True, dataloader_kwargs={}, **kwargs)
        Compute semantic embeddings for specified layers.
    search(text_input, templates=None, topk=10, threshold=0.0)
        Search for components in the concept database using text input.
    label(text_input, templates=None, concept_db=None, device=None)
        Label model components with text inputs.
    eval_clarity(concept_db=None)
        Evaluate concept clarity scores.
    eval_redundancy(concept_db=None)
        Evaluate concept redundancy scores.
    eval_polysemanticity(concept_db=None)
        Evaluate concept polysemanticity scores.

    Examples
    --------
    >>> import semanticlens as sl
    >>> # Initialize components
    >>> fm = sl.foundation_models.OpenClip("hf-hub:apple/MobileCLIP-S2-OpenCLIP")
    >>> cv = sl.component_visualization.ActivationComponentVisualizer(model, dataset, layer_names)
    >>> # Create lens
    >>> lens = sl.Lens(dataset, cv, fm, "imagenet", "cache")
    >>> # Compute embeddings
    >>> lens.compute_semantic_embeddigs(layer_names)
    >>> # Search and analyze
    >>> results, alignment = lens.search("red car")
    >>> clarity_scores = lens.eval_clarity()
    """

    def __init__(
        self,
        dataset: Dataset,
        component_visualizer: AbstractComponentVisualizer,
        foundation_model: VisionLanguageFoundationModel,
        dataset_name: str,
        storage_dir: str | Path,
        device=None,
    ):
        """
        Initialize a Lens instance for semantic analysis.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset containing input examples for analysis.
        component_visualizer : AbstractComponentVisualizer
            Visualizer for analyzing neural network components.
        foundation_model : VisionLanguageFoundationModel
            Foundation model for encoding images and text.
        dataset_name : str
            Name identifier for the dataset.
        storage_dir : str or Path
            Directory for caching computed embeddings and results.
        device : torch.device or str, optional
            Device for computation. If None, uses the foundation model's device.
        """
        self.dataset = dataset
        self.cv = component_visualizer
        self.fm = foundation_model
        self.fm_preprocessor = self.fm.processor
        self.layer_names = component_visualizer.layer_names
        self.dataset_name = dataset_name
        self.device = device or next(self.fm.parameters()).device
        self.storage_dir = storage_dir
        self._concept_db = None

    @property
    def concept_db(self):
        """
        Get the concept database containing computed embeddings.

        Returns
        -------
        dict
            Dictionary mapping layer names to ConceptTensor objects
            containing embeddings for each component.

        Raises
        ------
        ValueError
            If the concept database has not been initialized yet.
        """
        if self._concept_db is None:
            raise ValueError("Concept database not initialized. Call compute_semantic_embeddigs first.")
        return self._concept_db

    @concept_db.setter
    def concept_db(self, value):
        """
        Set the concept database.

        Parameters
        ----------
        value : dict
            Dictionary mapping layer names to ConceptTensor objects.

        Raises
        ------
        ValueError
            If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Concept database must be a dictionary.")
        self._concept_db = value

    def embed_text_dataset(
        self, texts: list | str, templates: list = ["an image of {}"], batch_size=32, cache=True, device=None
    ):
        """
        Embed a list of text concepts with optional templates.

        This method takes text inputs and embeds them using the foundation model,
        optionally applying templates to format the text appropriately. Results
        are cached for efficient reuse.

        Parameters
        ----------
        texts : list of str or str
            List of text concepts to embed, or single text string.
        templates : list of str, default=["an image of {}"]
            Template strings where {} will be replaced with each text.
            If None, no templates are applied.
        batch_size : int, default=32
            Batch size for processing text embeddings.
        cache : bool, default=True
            Whether to use caching for computed embeddings.
        device : torch.device or str, optional
            Device for computation. If None, uses the lens device.

        Returns
        -------
        tuple
            (ConceptTensor, numpy.ndarray): Embedded text concepts and text labels.

        Raises
        ------
        ValueError
            If texts is not a list, string, or numpy array.

        Examples
        --------
        >>> texts = ["cat", "dog", "bird"]
        >>> embeddings, labels = lens.embed_text_dataset(texts)
        >>> embeddings.shape
        torch.Size([3, 512])
        """
        # TODO create decorator for caching
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            raise ValueError("texts must be a list or a string")
        texts = sorted(texts)
        metadata = {"texts": texts, "templates": templates, "foundation_model": self.fm.name}

        if cache:
            hash_val = consistent_hash(json.dumps(metadata, sort_keys=True))
            fpath_ = Path(self.storage_dir) / "text_embeds" / self.fm.name / f"{hash_val}.safetensors"
            if fpath_.exists():
                text_embeds = sl.ConceptTensor.load(fpath_)
                if (
                    text_embeds.metadata["foundation_model"] == self.fm.name
                    and text_embeds.metadata["texts"] == texts
                    and text_embeds.metadata["templates"] == templates
                ):
                    logger.debug("Loaded from cache.")
                    return text_embeds, texts
                else:
                    logger.debug("Cache mismatch, recomputing!")
            else:
                logger.debug("Cache not found (%s)", fpath_)

        labels = [t.format(text) for t in templates or ["{}"] for text in texts]

        if templates:
            empty = [t.format("") for t in templates]
            empty_embeds = self.embed_text(empty, templates=None, device=device).cpu()

        text_embeds = []
        for i in tqdm(range(0, len(labels), batch_size), desc="Embedding texts"):
            batch = labels[i : i + batch_size]
            text_embed = self.embed_text(batch, templates=None, device=device).cpu()
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)

        if templates is None:
            if cache:  # TODO add proper caching
                fpath_.parent.mkdir(parents=True, exist_ok=True)
                sl.ConceptTensor(text_embeds, metadata=metadata).save(fpath_)
            return text_embeds, texts

        text_embeds = (
            einops.rearrange(
                text_embeds, "(d_temp d_txt) d_emb -> d_txt d_temp d_emb", d_txt=len(texts), d_temp=len(templates)
            )
            - empty_embeds[None]  # remove empty templates ...
        ).mean(1)  # ... and average over them

        # [ ] TODO do not normalize?
        text_embeds = normalize(text_embeds, p=2, dim=-1)
        text_embeds = sl.ConceptTensor(text_embeds, metadata=metadata)

        if cache:
            fpath_.parent.mkdir(parents=True, exist_ok=True)

            text_embeds.save(fpath_)

        return text_embeds, np.array(texts)

    @torch.no_grad()
    def embed_text(self, text, templates=None, device=None):
        """
        Embed a single text prompt with optional templates.

        This method encodes text using the foundation model, optionally applying
        templates to format the text appropriately. Template embeddings are
        corrected by subtracting empty template embeddings.

        Parameters
        ----------
        text : str
            Text to embed.
        templates : list of str or str, optional
            Template strings where {} will be replaced with the text.
            If None, no templates are applied.
        device : torch.device or str, optional
            Device for computation. If None, uses the lens device.

        Returns
        -------
        torch.Tensor
            Text embedding tensor.

        Raises
        ------
        ValueError
            If templates is not a list or string.

        Examples
        --------
        >>> embedding = lens.embed_text("cat", templates=["a photo of {}"])
        >>> embedding.shape
        torch.Size([1, 512])
        """
        device = device or self.device
        self.fm.to(device)
        if not templates:
            return self.fm.encode_text(**self.fm.processor(text=text)).cpu()

        if isinstance(templates, str):
            templates = [templates]
        elif not isinstance(templates, list):
            raise ValueError("templates must be a list or a string")

        text = [t.format(text) for t in templates]
        text_embed = self.fm.encode_text(**self.fm.processor(text=text)).cpu()

        empty = [t.format("") for t in templates]
        empty_embed = self.fm.encode_text(**self.fm.processor(text=empty)).cpu()

        # correct embedding
        text_embed = text_embed - empty_embed

        self.fm.to(self.device)
        return text_embed.mean(0).unsqueeze(0)

    @torch.no_grad()
    @cache_image_dataset_embeddings
    def embed_image_dataset(self, batch_size=32, device=None, cache=True, **dataloader_kwargs) -> sl.ConceptTensor:
        """
        Embed an image dataset using the foundation model.

        This method processes the entire dataset to compute visual embeddings for
        all images using the foundation model. Results are cached for efficient
        reuse and include metadata about the model and dataset.

        Parameters
        ----------
        batch_size : int, default=32
            Batch size for processing the dataset.
        device : torch.device or str, optional
            Device for computation. If None, uses the lens device.
        cache : bool, default=True
            Whether to use caching for computed embeddings.
        **dataloader_kwargs
            Additional keyword arguments passed to the DataLoader.

        Returns
        -------
        ConceptTensor
            Tensor containing embeddings for all images in the dataset
            with associated metadata.

        Examples
        --------
        >>> embeddings = lens.embed_image_dataset(batch_size=64)
        >>> embeddings.shape
        torch.Size([10000, 512])  # 10000 images, 512-dim embeddings
        >>> embeddings.metadata
        {'foundation_model': 'hf_hub_apple_MobileCLIP_S2_OpenCLIP', 'dataset': 'imagenet', 'ids': [0, 1, ...]}
        """
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset_ = self.dataset.dataset
            data_ids = self.dataset.indices.tolist()
        else:
            dataset_ = self.dataset
            data_ids = list(range(len(dataset_)))
        # [ ] TODO avoid inplace operation due to side-effects if possible
        dataset_.transform = lambda x: self.fm.processor(images=x)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            **dataloader_kwargs,
        )

        device = device or self.device
        self.fm.to(device)

        embeds = []
        with tqdm(total=len(self.dataset), desc="Embedding dataset...") as pbar_dataset:
            for batch in dataloader:
                data = batch[0] if isinstance(batch, (tuple, list)) else batch
                data = data.to(device)  # TODO optimize proprocessing for speed
                fm_out = self.fm.encode_vision(**data).cpu()
                embeds.append(fm_out)
                pbar_dataset.update(batch_size)

        dataset_embeds = sl.ConceptTensor(
            torch.cat(embeds, dim=0),
            metadata={"foundation_model": self.fm.name, "dataset": self.dataset_name, "ids": data_ids},
        )

        self.fm.to(self.device)
        return dataset_embeds

    def _get_dataset_and_ids(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            return self.dataset.dataset, self.dataset.indices.tolist()
        return self.dataset, list(range(len(self.dataset)))

    def compute_semantic_embeddigs(
        self,
        layer_names: list[str] | str,
        component_specific_examples: bool = False,
        batch_size: int = 32,
        device: str = None,
        cache=True,
        dataloader_kwargs: dict = {},
        **kwargs,
    ) -> dict[str, sl.ConceptTensor]:
        """Compute semantic embeddings for the specified layers.

        This method takes a set of layer names and computes semantic embeddings for each layer.
        Two modes are supported:
        - "full_input": First embeds each input sample, then constructs specific embeddings
        - "component_specific": Directly computes embeddings for specific components

        Args:
            layer_names: List of layer names or a single layer name string
            example_mode: Mode to compute embeddings ("full_input" or "component_specific")
            batch_size: Batch size for processing
            device: Device to use for computation
            cache: Whether to use cached embeddings
            dataloader_kwargs: Additional arguments to pass to the DataLoader
            **kwargs: Additional arguments for component-specific embedding computation

        Returns
        -------
            Dictionary mapping layer names to their semantic embeddings as sl.ConceptTensors

        """
        if component_specific_examples:
            return self.compute_component_specific_embeddings(
                layer_names=layer_names,
                batch_size=batch_size,
                device=device,
                cache=cache,
                dataloader_kwargs=dataloader_kwargs,
            )

        # full_input -> first embed each input sample -> second construct specific embeddings
        # TODO could be improved by only embedding the necessary images (too much overhead for now)
        dataset_embeddings = self.embed_image_dataset(
            batch_size=batch_size,
            device=device,
            cache=cache,
            **dataloader_kwargs,
        )
        self.concept_db = {}
        for layer_name in layer_names:
            self.concept_db[layer_name] = dataset_embeddings[self.cv.get_act_max_sample_ids(layer_name)]
        return self.concept_db

    def compute_component_specific_embeddings(
        self, layer_names, batch_size, composite=None, n_ref=None, rf=True, cache=True, **kwargs
    ) -> dict[str, sl.ConceptTensor]:
        """
        Compute embeddings using component-specific examples.

        This method computes semantic embeddings by using examples that maximally
        activate specific components, rather than using the full input dataset.
        It processes each layer separately and caches results for efficiency.

        Parameters
        ----------
        layer_names : list of str
            Names of the layers to compute embeddings for.
        batch_size : int
            Batch size for processing components.
        composite : optional
            Composite rule for attribution methods (used with relevance-based visualizers).
        n_ref : int, optional
            Number of reference examples to use per component. If None, uses all available.
        rf : bool, default=True
            Whether to use receptive field information.
        cache : bool, default=True
            Whether to use caching for computed embeddings.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict[str, ConceptTensor]
            Dictionary mapping layer names to ConceptTensor objects containing
            component-specific embeddings.

        Examples
        --------
        >>> concept_db = lens.compute_component_specific_embeddings(
        ...     layer_names=["layer3", "layer4"],
        ...     batch_size=32,
        ...     n_ref=10
        ... )
        >>> concept_db["layer3"].shape
        torch.Size([512, 10, 512])  # 512 components, 10 refs each, 512-dim embeddings
        """
        # TODO add caching!

        concept_db = {}
        logger.info("Computing semantic embeddings using component specific examples...")
        pbar_layer = tqdm(total=len(layer_names), desc="Layers")
        for layer_name in layer_names:
            pbar_layer.set_description(f'Layer "{layer_name}"')
            concept_db[layer_name] = self._compute_component_specific_embeddings_layer(
                batch_size, composite, n_ref, rf, layer_name, cache
            )
            pbar_layer.update(1)

        self.cv.to(self.device)
        pbar_layer.close()

        self.concept_db = concept_db
        return self.concept_db

    @cache_component_specific_embeddings
    def _compute_component_specific_embeddings_layer(self, batch_size, composite, n_ref, rf, layer_name, cache):
        n_components, n_ref_ = self.cv.get_act_max_sample_ids(layer_name).shape
        n_ref = n_ref or n_ref_
        component_ids = torch.arange(n_components)
        layer_embeddings = []

        pbar_components = tqdm(total=n_components, desc="Components", leave=False)
        for batch_id in range(0, len(component_ids), batch_size):
            current_ids = component_ids[batch_id : batch_id + batch_size]

            self.fm.to("cpu")
            self.cv.to(self.device)

            pbar_components.set_description("Components [collecting component refs...]")

            concept_example_dict = self.cv.get_max_reference(
                concept_ids=current_ids.tolist(),
                layer_name=layer_name,
                n_ref=n_ref,
                batch_size=n_ref,  # TODO fix bug in zennit-crp
            )
            pbar_components.set_description("Components")

            self.cv.to("cpu")
            self.fm.to(self.device)

            concept_examples_pil = [ex for cpt_exs in concept_example_dict.values() for ex in cpt_exs]

            with torch.no_grad():
                pbar_embed = tqdm(total=len(concept_examples_pil), desc="Embedding", leave=False)

                embeddings = []
                for pil_batch_id in range(0, len(concept_examples_pil), batch_size):
                    pil_batch = concept_examples_pil[pil_batch_id : pil_batch_id + batch_size]

                    embeddings.append(self.fm.encode_vision(**self.fm.processor(images=pil_batch)).cpu())

                    pbar_embed.update(len(pil_batch))
                pbar_embed.close()

            layer_embeddings.append(torch.cat(embeddings).reshape(batch_size, n_ref, -1))

            pbar_components.update(batch_size)
        pbar_components.close()

        layer_embeddings = sl.ConceptTensor(
            torch.cat(layer_embeddings, dim=0),
            metadata={
                "layer_name": layer_name,
                "foundation_model": self.fm.name,
                "dataset": self.dataset_name,
                "ids": component_ids.tolist(),
                "n_reference_samples": n_ref,
                "component_visualizer": json.dumps(self.cv.metadata, sort_keys=True),  # TODO create this!
            },
        )
        return layer_embeddings

    def search(
        self, text_input: sl.ConceptTensor | list[str] | str, templates: None | list[str] = None, topk=10, threshold=0.0
    ):
        """
        Search for components in the concept database using text input.

        This method searches through the concept database to find neural network
        components that are most aligned with the given text input. It returns
        the top-k most similar components for each layer.

        Parameters
        ----------
        text_input : ConceptTensor or list of str or str
            Text input to search with. Can be embeddings, list of strings, or single string.
        templates : list of str, optional
            Template strings to format text input. Only used if text_input is string(s).
        topk : int, default=10
            Number of top components to return per layer.
        threshold : float, default=0.0
            Minimum similarity threshold for returned components.

        Returns
        -------
        tuple
            (results, alignment):
            - results: dict mapping layer names to component IDs above threshold
            - alignment: dict mapping layer names to full alignment scores

        Examples
        --------
        >>> results, alignment = lens.search("red car", topk=5)
        >>> results["layer4"]  # Top 5 component IDs in layer4
        tensor([42, 17, 89, 156, 203])
        >>> alignment["layer4"].shape  # Full alignment scores
        torch.Size([512])  # 512 components in layer4
        """
        if not isinstance(text_input, sl.ConceptTensor):
            text_input = self.embed_text(text_input, templates=templates, device=self.device)

        alignment = label(text_embeds=text_input, concept_db=self.concept_db, device=self.device)

        results = {}
        for layer_name, align_scores in alignment.items():
            vals, ids = align_scores.topk(topk, dim=0)  # first dim corresponds to components
            results[layer_name] = ids[vals > threshold]
        return results, alignment

    def label(
        self,
        text_input: sl.ConceptTensor | list[str] | str,
        templates: None | list[str] = None,
        concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None,
        device=None,
    ):
        """
        Label model components with text inputs.

        This method computes alignment scores between text embeddings and
        concept embeddings to determine how well text labels describe each
        neural network component.

        Parameters
        ----------
        text_input : ConceptTensor or list of str or str
            Text input for labeling. Can be embeddings, list of strings, or single string.
        templates : list of str, optional
            Template strings to format text input. Only used if text_input is string(s).
        concept_db : dict or ConceptTensor, optional
            Concept database to use. If None, uses the lens's concept database.
        device : torch.device or str, optional
            Device for computation. If None, uses the lens device.

        Returns
        -------
        dict or tuple
            If text_input is not ConceptTensor: (alignment_scores, text_labels)
            If text_input is ConceptTensor: alignment_scores only
            where alignment_scores maps layer names to similarity scores.

        Examples
        --------
        >>> alignment, labels = lens.label(["cat", "dog", "bird"])
        >>> alignment["layer4"].shape
        torch.Size([3, 512])  # 3 labels x 512 components
        >>> labels
        array(['cat', 'dog', 'bird'])
        """
        if concept_db is None:
            concept_db = self.concept_db
        device = device or self.device

        if not isinstance(text_input, sl.ConceptTensor):
            text_input, labels = self.embed_text_dataset(texts=text_input, templates=templates, device=device)

            return label(text_embeds=text_input, concept_db=concept_db, device=device), labels

        return label(text_embeds=text_input, concept_db=concept_db, device=device)

    def eval_clarity(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """
        Evaluate concept clarity scores.

        Computes clarity scores for all concepts to measure how uniform
        the concept examples are, indicating representation clarity.

        Parameters
        ----------
        concept_db : dict or ConceptTensor, optional
            Concept database to evaluate. If None, uses the lens's concept database.

        Returns
        -------
        dict or torch.Tensor
            If concept_db is dict: dict mapping layer names to clarity score tensors.
            If concept_db is ConceptTensor: single clarity score tensor.
            Higher values indicate clearer concepts.

        See Also
        --------
        semanticlens.scores.clarity_score : Detailed documentation of clarity computation.

        Examples
        --------
        >>> clarity_scores = lens.eval_clarity()
        >>> clarity_scores["layer4"].shape
        torch.Size([512])  # 512 components in layer4
        """
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.clarity_score(concept_db)

        return {
            layer_name: sl.clarity_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(concept_db.items(), leave=False, desc="Evaluating clarity scores")
        }

    def eval_redundancy(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """
        Evaluate concept redundancy scores.

        Computes redundancy scores to measure how redundant concepts are
        across different neural components.

        Parameters
        ----------
        concept_db : dict or ConceptTensor, optional
            Concept database to evaluate. If None, uses the lens's concept database.

        Returns
        -------
        dict or torch.Tensor
            If concept_db is dict: dict mapping layer names to redundancy score tensors.
            If concept_db is ConceptTensor: single redundancy score tensor.
            Higher values indicate more redundant concepts.

        See Also
        --------
        semanticlens.scores.redundancy_score : Detailed documentation of redundancy computation.

        Examples
        --------
        >>> redundancy_scores = lens.eval_redundancy()
        >>> redundancy_scores["layer4"].shape
        torch.Size([512])  # 512 components in layer4
        """
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.redundancy_score(concept_db)

        return {
            layer_name: sl.redundancy_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(concept_db.items(), leave=False, desc="Evaluating redundancy scores")
        }

    def eval_polysemanticity(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """
        Evaluate concept polysemanticity scores.

        Computes polysemanticity scores to measure how polysemantic (multi-meaning)
        concepts are using clustering analysis.

        Parameters
        ----------
        concept_db : dict or ConceptTensor, optional
            Concept database to evaluate. If None, uses the lens's concept database.

        Returns
        -------
        dict or torch.Tensor
            If concept_db is dict: dict mapping layer names to polysemanticity score tensors.
            If concept_db is ConceptTensor: single polysemanticity score tensor.
            Higher values indicate more polysemantic concepts.

        See Also
        --------
        semanticlens.scores.polysemanticity_score : Detailed documentation of polysemanticity computation.

        Examples
        --------
        >>> poly_scores = lens.eval_polysemanticity()
        >>> poly_scores["layer4"].shape
        torch.Size([512])  # 512 components in layer4
        """
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.polysemanticity_score(concept_db)

        return {
            layer_name: sl.polysemanticity_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(
                concept_db.items(), leave=False, desc="Evaluating polysemanticity scores"
            )
        }


@torch.no_grad()
def label(text_embeds: sl.ConceptTensor, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor, device=None):
    """
    Compute alignment of text embeddings with concept embeddings.

    This function computes cosine similarity between text embeddings and
    concept embeddings to determine how well text labels align with
    neural network components.

    Parameters
    ----------
    text_embeds : ConceptTensor
        Text embeddings to align with concept embeddings.
    concept_db : dict or ConceptTensor
        Concept database containing embeddings for neural components.
        If dict, maps layer names to ConceptTensor objects.
        If ConceptTensor, single tensor of concept embeddings.
    device : torch.device or str, optional
        Device for computation. If None, uses CUDA if available, else CPU.

    Returns
    -------
    torch.Tensor or dict
        If concept_db is ConceptTensor: returns alignment tensor of shape
        (n_concepts, n_text_embeddings).
        If concept_db is dict: returns dict mapping layer names to alignment tensors.

    Examples
    --------
    >>> text_embeds = lens.embed_text_dataset(["cat", "dog"])
    >>> alignment = label(text_embeds, lens.concept_db)
    >>> alignment["layer4"].shape
    torch.Size([512, 2])  # 512 components x 2 text labels
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(concept_db, sl.ConceptTensor):
        alignment = (normalize(concept_db.mean(1), dim=-1).to(device) @ text_embeds.T.to(device)).cpu()
        return alignment
    result = {}
    for layer_name, concept_embeds in concept_db.items():
        result[layer_name] = (normalize(concept_embeds.mean(1), dim=-1).to(device) @ text_embeds.T.to(device)).cpu()
    return result
