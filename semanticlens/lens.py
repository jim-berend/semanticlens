"""
Lens: Main class for visual concept analysis and exploration.

This module provides the primary interface for semantic analysis of neural networks,
combining component visualization with foundation models to explore relationships
between visual concepts and text embeddings.
"""

from __future__ import annotations

import logging

import einops
import PIL
import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.foundation_models.base import AbstractVLM
from semanticlens.scores import clarity_score, polysemanticity_score, redundancy_score, similarity_score
from semanticlens.utils.helper import get_fallback_name

logger = logging.getLogger(__name__)


def compute_concept_db(cv: AbstractComponentVisualizer, fm: AbstractVLM):
    """Compute a concept database in a stateless manner.

    This function delegates the computation of the concept database to the
    provided component visualizer instance. It follows an Inversion of Control
    (IoC) pattern where the visualizer, which holds the logic for extracting
    concepts, is controlled by this function to perform the embedding using the
    provided foundation model.

    Parameters
    ----------
    cv : AbstractComponentVisualizer
        An initialized component visualizer instance (e.g.,
        `ActivationComponentVisualizer`) that has already been run to find
        concept examples.
    fm : AbstractVLM
        An initialized foundation model instance (e.g., `OpenClip`) used for
        embedding the concept examples.
    **kwargs
        Additional keyword arguments to be passed to the component visualizer's
        internal computation method.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary mapping layer names to their corresponding concept
        database. Each concept database is a tensor of shape
        (n_components, n_samples, embedding_dim).
    """
    return cv._compute_concept_db(fm)


def text_probing(
    fm: AbstractVLM,
    query: str | list[str],
    aggregated_concept_db: torch.Tensor | dict[str, torch.Tensor],
    templates: list[str] | None = None,
    batch_size: int | None = None,
):
    """Probe a concept database with text queries to find matching concepts.

    This function searches for concepts in a model's learned representations
    using natural language. It works by embedding the text query using the
    foundation model and then computing the cosine similarity between the query
    embedding and the concept embeddings in the database.

    Parameters
    ----------
    fm : AbstractVLM
        An initialized foundation model instance for encoding the text query.
    query : str or list[str]
        The text query or a list of text queries to search for.
    aggregated_concept_db : torch.Tensor or dict[str, torch.Tensor]
        The aggregated concept database to search within. This should contain
        a single embedding for each concept (e.g. mean aggregated),
        resulting in a tensor of shape (n_components, embedding_dim).
        It can be a single tensor or a dictionary mapping layer names to tensors.
    templates : list[str], optional
        A list of prompt templates, e.g., "a photo of {}". Using templates can
        improve search fidelity by averaging out the influence of the prompt's
        structure. If provided, the embedding of an empty template is subtracted
        from the embedding of the filled template.
    batch_size : int, optional
        The batch size for embedding text queries if multiple templates are used.
        If None, all templated queries are processed in one batch.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        A tensor or a dictionary of tensors containing the cosine similarity
        scores between the query embedding(s) and the concept embeddings. Higher
        scores indicate a closer semantic match.

    Examples
    --------
    >>> import torch
    >>> from semanticlens.foundation_models.clip import OpenClip
    >>> from semanticlens.lens import text_probing
    >>>
    >>> # Mock foundation model and concept database
    >>> fm = OpenClip(url="...")
    >>> lens = ConceptLens(fm)
    >>> concepts = lens.compute_concept_db(cv)
    >>> # Find neurons related to "dogs"
    >>> scores = text_probing(fm=fm,query="dog", aggregated_concept_db=concepts, templates=["a photo of a {}"])
    >>> top_neuron = scores["layer4"].argmax()
    >>> print(f"Top matching neuron for 'dog': {top_neuron.item()}")
    """
    queries = query if isinstance(query, list) else [query]
    query_embeds = _embed_text_probes(fm, queries, templates, batch_size)

    assert query_embeds.ndim == 2
    assert query_embeds.shape[0] == len(queries)

    return _probe(query_embeds, aggregated_concept_db)


def image_probing(
    fm: AbstractVLM,
    query: PIL.Image | list[PIL.Image],
    aggregated_concept_db: torch.Tensor | dict[str, torch.Tensor],
):
    """Probe a concept database with image queries to find matching concepts.

    This function searches for concepts that are semantically similar to a given
    query image or a set of query images. It embeds the image(s) using the
    foundation model and computes the cosine similarity against the concept embeddings in the database.

    If a list of images is provided, their embeddings are averaged to form a
    single probe vector. This is useful for finding concepts that represent the
    common theme across multiple images.

    Parameters
    ----------
    fm : AbstractVLM
        An initialized foundation model instance for encoding the image query.
    query : PIL.Image.Image or list[PIL.Image.Image]
        A single PIL image or a list of PIL images to use as the query.
    aggregated_concept_db : torch.Tensor or dict[str, torch.Tensor]
        The aggregated concept database to search within. This should contain
        the mean embedding for each concept, resulting in a tensor of shape
        (n_components, embedding_dim). It can be a single tensor or a
        dictionary mapping layer names to tensors.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        A tensor or a dictionary of tensors containing the cosine similarity
        scores between the image query embedding and the concept embeddings.
        Higher scores indicate a closer semantic match.
    """
    with torch.no_grad():
        query_embed = fm.encode_image(fm.preprocess(query).to(fm.device)).cpu()
    query_embed = query_embed.mean(0)[None] if query_embed.shape[0] > 1 else query_embed

    return _probe(query_embed, aggregated_concept_db)


@torch.no_grad()
def _embed_text_probes(
    fm: AbstractVLM,
    query: list[str],
    templates: list[str] | None,
    batch_size: int | None,
):
    """Templating and embedding logic of text-probes."""
    if templates:
        query_templated = [t.format(q) for t in templates for q in query]
        empty_templates = [t.format("") for t in templates]

        batch_size = batch_size or len(query_templated)

        query_templated_embeds = list()
        for batch_idx in tqdm(
            range(0, len(query_templated), batch_size),
            desc="text embedding ...",
            leave=False,
            disable=batch_size == len(query_templated),
        ):
            query_templated_batch = query_templated[batch_idx : batch_idx + batch_size]
            query_templated_embeds.append(
                fm.encode_text(fm.tokenize(query_templated_batch).to(fm.device)).cpu()  # handle device in tokenization?
            )
        query_templated_embeds = torch.cat(query_templated_embeds, dim=0)

        empty_templates_embeds = fm.encode_text(
            fm.tokenize(empty_templates).to(fm.device)
        ).cpu()  # handle device in tokenization?

        query_embed = (
            einops.rearrange(query_templated_embeds, "(q t) d -> q t d", q=len(query))
            - einops.rearrange(empty_templates_embeds, "t d -> 1 t d")
        ).mean(1)

    else:
        query_embed = fm.encode_text(fm.tokenize(query).to(fm.device)).cpu()
    return query_embed


@torch.no_grad()
def _probe(
    query: torch.Tensor, aggregated_concept_db: torch.tensor | dict[str, torch.Tensor]
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(aggregated_concept_db, torch.Tensor):
        tensors = similarity_score(query.to(aggregated_concept_db.device), aggregated_concept_db)
    else:
        tensors = {key: similarity_score(query.to(value.device), value) for key, value in aggregated_concept_db.items()}
    return tensors


class Lens:
    """Orchestration layer for feature extraction and concept probing.

    The `Lens` class is the main entry point for using the `semanticlens`
    library. It provides a high-level, stateful interface that manages a
    foundation model and orchestrates the entire semantic analysis workflow,
    from computing a concept database to searching it and evaluating its
    interpretability.

    This class simplifies the process by holding the state of the foundation
    model and providing convenient methods that wrap the core functionalities of
    the package.

    Parameters
    ----------
    fm : AbstractVLM
        An initialized vision-language foundation model that will be used for
        all embedding and probing tasks.
    device : str or torch.device, optional
        The device to run the foundation model on (e.g., "cuda", "cpu"). If None,
        the model's current device is used.

    Attributes
    ----------
    fm : AbstractVLM
        The foundation model instance used by the Lens.
    device : torch.device
        The device on which the foundation model is located.

    Examples
    --------
    >>> import torch
    >>> from semanticlens import Lens
    >>> from semanticlens.foundation_models import ClipMobile
    >>> from semanticlens.component_visualization import ActivationComponentVisualizer
    >>>
    >>> # 1. Initialize the Lens with a foundation model
    >>> fm = ClipMobile(device="cpu")
    >>> lens = Lens(fm=fm)
    >>>
    >>> # 2. Assume `cv` is an initialized ActivationComponentVisualizer
    >>> cv = ActivationComponentVisualizer(...)
    >>>
    >>> # 3. Compute the concept database
    >>> concept_db = lens.compute_concept_db(cv)
    >>>
    >>> # 4. Probe the database with a text query
    >>> aggregated_db = {"layer4": concept_db["layer4"].mean(dim=1)}
    >>> scores = lens.text_probing("a photo of a cat", aggregated_db)

    """

    def __init__(self, fm: str | AbstractVLM, device=None):
        self.fm: AbstractVLM = fm
        self.device = device or self.fm.device
        self.fm.to(self.device)

        if not hasattr(self.fm, "name"):
            self.fm.name = get_fallback_name(self.fm)
            logger.debug(f"Assigned fallback name to foundation model: {self.fm.name}")

    def compute_concept_db(self, cv: AbstractComponentVisualizer, **kwargs) -> dict[str, torch.Tensor]:
        """Compute or load from cache the concept database for a visualizer.

        This method orchestrates the creation of the concept database, which is a
        semantic representation of the concepts learned by a model's components.
        It follows an Inversion of Control (IoC) pattern by calling the internal
        `_compute_concept_db` method of the provided component visualizer `cv`.

        If caching is enabled in the component visualizer, this method will first
        attempt to load the concept database from a pre-computed cache file. If
        the file does not exist, it will compute the database and save it to the
        cache for future use.

        Parameters
        ----------
        cv : AbstractComponentVisualizer
            An initialized component visualizer that has already collected the
            maximally activating samples for the target model's components
            (i.e., `cv.run()` has been called).
        **kwargs
            Additional keyword arguments to be passed to the visualizer's
            `_compute_concept_db` method, such as `batch_size` or `num_workers`.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary mapping layer names to their concept databases. Each
            database is a tensor of shape (n_components, n_samples,
            embedding_dim).
        """
        if cv.caching:
            fdir = cv.storage_dir / "concept_database" / self.fm.name
            fdir.mkdir(parents=True, exist_ok=True)
            fname = (
                "concept_db-"
                + "-".join([v for k, v in cv.metadata.items() if k not in ["dataset", "model"]])
                + ".safetensors"
            )
            fpath = fdir / fname
            if fpath.exists():
                logger.debug("Loading concept DB from cache")
                return load_file(filename=fpath)
            logger.debug("Computing concept DB and saving to cache")
            concept_db = cv._compute_concept_db(self.fm, **kwargs)
            save_file(tensors=concept_db, filename=fpath)
            logger.debug(f"Saved concept DB to cache {fpath}")

            return concept_db

        else:
            logger.debug("Caching is not enabled. Computing Concept DB")
            return cv._compute_concept_db(self.fm, **kwargs)

    def text_probing(
        self,
        query: str | list[str],
        aggregated_concept_db: torch.Tensor | dict[str, torch.Tensor],
        templates: list[str] | None = None,
        batch_size: int | None = None,
    ):
        """Probe a concept database with text queries to find matching concepts.

        This method is a convenient wrapper around the stateless
        :func:`~text_probing` function. It uses the foundation model stored
        within the `Lens` instance to perform the search.

        Parameters
        ----------
        query : str or list[str]
            The text query or a list of text queries to search for.
        aggregated_concept_db : torch.Tensor or dict[str, torch.Tensor]
            The aggregated concept database to search within, with tensors of
            shape (n_components, embedding_dim).
        templates : list[str], optional
            A list of prompt templates, e.g., "a photo of {}".
        batch_size : int, optional
            The batch size for embedding text queries.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A tensor or a dictionary of tensors containing the cosine similarity
            scores between the query and the concepts.
        """
        return text_probing(self.fm, query, aggregated_concept_db, templates, batch_size)

    def image_probing(
        self,
        query: PIL.Image | list[PIL.Image],
        aggregated_concept_db: torch.Tensor | dict[str, torch.Tensor],
    ):
        """Probe a concept database with image queries to find matching concepts.

        This method is a convenient wrapper around the stateless
        :func:`~image_probing` function. It uses the foundation model stored
        within the `Lens` instance to perform the search.

        Parameters
        ----------
        query : PIL.Image.Image or list[PIL.Image.Image]
            A single PIL image or a list of PIL images to use as the query.
        aggregated_concept_db : torch.Tensor or dict[str, torch.Tensor]
            The aggregated concept database to search within, with tensors of
            shape (n_components, embedding_dim).

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A tensor or a dictionary of tensors containing the cosine similarity
            scores between the query and the concepts.
        """
        return image_probing(self.fm, query, aggregated_concept_db)

    def eval_clarity(self, concept_db: torch.Tensor | dict[str, torch.Tensor]):
        """Compute the clarity score for concepts in the database.

        Clarity measures how semantically coherent the examples for each concept
        are. A high clarity score suggests that a neuron has learned a
        well-defined, easily understandable concept.

        This method wraps the :func:`~semanticlens.scores.clarity_score` function.

        Parameters
        ----------
        concept_db : torch.Tensor or dict[str, torch.Tensor]
            A concept database tensor of shape (n_components, n_samples,
            embedding_dim), or a dictionary mapping layer names to such tensors.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A tensor of clarity scores for each component, or a dictionary of
            such tensors.

        See Also
        --------
        semanticlens.scores.clarity_score : The underlying function for the score.
        """
        if isinstance(concept_db, torch.Tensor):
            return clarity_score(concept_db)
        else:
            return {key: clarity_score(value) for key, value in concept_db.items()}

    def eval_redundancy(self, aggregated_concept_db: torch.Tensor | dict[str, torch.Tensor]):
        """Compute the redundancy score for concepts in the database.

        Redundancy measures the degree of semantic overlap between different
        components (e.g. neurons) in a layer. It is calculated as the average maximal
        similarity of each component to any other component in the set.

        This method wraps the :func:`~semanticlens.scores.redundancy_score` function.

        Parameters
        ----------
        aggregated_concept_db : torch.Tensor or dict[str, torch.Tensor]
            An aggregated concept database tensor of shape (n_components,
            embedding_dim), or a dictionary mapping layer names to such tensors.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A tensor representing the mean redundancy score, or a dictionary of
            such scores for each layer.

        See Also
        --------
        semanticlens.scores.redundancy_score : The underlying function for the score.
        """
        if isinstance(aggregated_concept_db, torch.Tensor):
            return redundancy_score(aggregated_concept_db)
        else:
            return {key: redundancy_score(value) for key, value in aggregated_concept_db.items()}

    def eval_polysemanticity(self, concept_db: torch.Tensor | dict[str, torch.Tensor]):
        """Compute the polysemanticity score for concepts in the database.

        Polysemanticity measures whether a single neuron encodes multiple,
        semantically distinct concepts. The score is calculated by clustering
        the examples for each concept and measuring the diversity of the
        resulting cluster centers.

        This method wraps the :func:`~semanticlens.scores.polysemanticity_score` function.

        Parameters
        ----------
        concept_db : torch.Tensor or dict[str, torch.Tensor]
            A concept database tensor of shape (n_components, n_samples,
            embedding_dim), or a dictionary mapping layer names to such tensors.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A tensor of polysemanticity scores for each component, or a dictionary
            of such tensors.

        See Also
        --------
        semanticlens.scores.polysemanticity_score : The underlying function for the score.
        """
        if isinstance(concept_db, torch.Tensor):
            return polysemanticity_score(concept_db)
        else:
            return {key: polysemanticity_score(value) for key, value in concept_db.items()}
