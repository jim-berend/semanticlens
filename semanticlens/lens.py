import functools
import hashlib
import json
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

# TODO Device

# helpers ---


def consistent_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def cache_image_dataset_embeddings(method):
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
                        print("Loaded from cache.")
                        return cached
                    else:
                        print("Cache mismatch, recomputing!")
                except Exception as e:
                    print("Caching mechanism failed:", e)
        result = method(self, *args, cache=False, **kwargs)

        if cache:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            result.save(fpath)
            print("Saved to cache at: ", fpath)

        return result

    return wrapper


def cache_component_specific_embeddings(method):
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
                        print("Loaded from cache.")
                        return cached[:, :n_ref]
                    else:
                        print("Cache mismatch, recomputing!")
                except Exception as e:
                    print("Caching mechanism failed:", e)

        result = method(self, batch_size, composite, n_ref, rf, layer_name, cache=False)

        if cache:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            result.save(fpath)
            print("Saved to cache at: ", fpath)

        return result

    return wrapper


# main ---


class Lens:
    """
    A class for visual concept analysis and exploration.

    Lens provides methods to embed images and text datasets for semantic analysis.
    It integrates feature visualization (fv) with foundation models (fm) to
    explore relationships between visual concepts and text embeddings.

    Methods:
        embed_text_dataset: Embeds a list of text concepts, applying templates if provided.
        embed_text: Embeds a single text prompt.
        embed_image_dataset: Embeds an image dataset using the foundation model.
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
        if self._concept_db is None:
            raise ValueError("Concept database not initialized. Call compute_semantic_embeddigs first.")
        return self._concept_db

    @concept_db.setter
    def concept_db(self, value):
        if not isinstance(value, dict):
            raise ValueError("Concept database must be a dictionary.")
        self._concept_db = value

    def embed_text_dataset(
        self, texts: list | str, templates: list = ["an image of {}"], batch_size=32, cache=True, device=None
    ):
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
                    print("Loaded from cache.")
                    return text_embeds, texts
                else:
                    print("Cache mismatch, recomputing!")
            else:
                print("Cache not found (", fpath_, ")")

        labels = [t.format(l) for t in templates or ["{}"] for l in texts]

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
        device = device or self.device
        self.fm.to(device)
        if not templates:
            return self.fm.encode_text(**self.fm.processor(text=text)).cpu()

        assert isinstance(templates, list)

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
        """
        Compute semantic embeddings for the specified layers.

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

        Returns:
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
        # TODO add caching!

        concept_db = {}
        print("Computing semantic embeddings using component specific examples...")
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
        """Search via a text input for components in the concept database of the model"""
        # TODO add proper docstring
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
        """Label the model components with the text inputs."""
        if concept_db is None:
            concept_db = self.concept_db
        device = device or self.device

        if not isinstance(text_input, sl.ConceptTensor):
            text_input, labels = self.embed_text_dataset(texts=text_input, templates=templates, device=device)

            return label(text_embeds=text_input, concept_db=concept_db, device=device), labels

        return label(text_embeds=text_input, concept_db=concept_db, device=device)

    def eval_clarity(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """Evaluate Concept Clarity - see `semanticlens.scores` for more details."""
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.clarity_score(concept_db)

        return {
            layer_name: sl.clarity_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(concept_db.items(), leave=False)
        }

    def eval_redundancy(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """Evaluate Concept Redundancy - see `semanticlens.scores` for more details."""
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.redundancy_score(concept_db)

        return {
            layer_name: sl.redundancy_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(concept_db.items(), leave=False)
        }

    def eval_polysemanticity(self, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor | None = None):
        """Evaluate Concept Polysemanticity - see `semanticlens.scores` for more details."""
        if concept_db is None:
            concept_db = self.concept_db

        if isinstance(concept_db, sl.ConceptTensor):
            return sl.polysemanticity_score(concept_db)

        return {
            layer_name: sl.polysemanticity_score(concept_embeds)
            for layer_name, concept_embeds in tqdm(concept_db.items(), leave=False)
        }


@torch.no_grad()
def label(text_embeds: sl.ConceptTensor, concept_db: dict[str, sl.ConceptTensor] | sl.ConceptTensor, device=None):
    """Compute alignment of text embeddings with concept embeddings"""
    # TODO add docstring
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(concept_db, sl.ConceptTensor):
        alignment = (normalize(concept_db.mean(1), dim=-1).to(device) @ text_embeds.T.to(device)).cpu()
        return alignment
    result = {}
    for layer_name, concept_embeds in concept_db.items():
        result[layer_name] = (normalize(concept_embeds.mean(1), dim=-1).to(device) @ text_embeds.T.to(device)).cpu()
    return result
