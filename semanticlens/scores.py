"""
Scoring functions for evaluating concept quality in semantic analysis.

This module provides various metrics to assess the quality and characteristics
of learned concepts in neural networks, including clarity, redundancy, and
polysemanticity scores.
"""

import logging

import numpy as np
import torch
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@torch.inference_mode()
def clarity_score(V):
    """
    Compute clarity score for concept representations.

    Measures how uniform the concept examples are, indicating how clear
    the representation is. Higher values indicate better clarity.

    Parameters
    ----------
    V : torch.Tensor
        Concept tensor of shape (n_neurons, n_samples, n_features).

    Returns
    -------
    torch.Tensor
        Clarity scores of shape (n_neurons,). Values in range [-1/(n_samples-1), 1],
        where higher values indicate clearer concepts.

    Examples
    --------
    >>> V = torch.randn(10, 20, 512)  # 10 neurons, 20 samples, 512 features
    >>> clarity = clarity_score(V)
    >>> clarity.shape
    torch.Size([10])
    """
    # V.shape = (n_neurons) x n_samples x n_features
    V_nrmed = torch.nn.functional.normalize(V, dim=-1)
    clarity = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity


@torch.inference_mode()
def redundancy_score(cones):
    """
    Compute redundancy score for concept representations.

    Measures the redundancy across neurons by computing pairwise similarities
    and taking the maximum similarity for each neuron.

    Parameters
    ----------
    cones : torch.Tensor
        Concept tensor of shape (n_neurons, n_features).

    Returns
    -------
    torch.Tensor
        Redundancy scores of shape (n_neurons,). Higher values indicate
        more redundant representations.

    Examples
    --------
    >>> cones = torch.randn(10, 512)  # 10 neurons, 512 features
    >>> redundancy = redundancy_score(cones)
    >>> redundancy.shape
    torch.Size([])
    """
    device = cones.device
    cones_nrmed = torch.nn.functional.normalize(cones, dim=-1)
    sims = torch.matmul(cones_nrmed, cones_nrmed.swapaxes(-1, -2))
    sims = sims - 2 * torch.eye(sims.shape[-1]).to(device)  # remove diagonal
    redundancy = sims.max(-1).values.mean(-1)
    return redundancy


@torch.inference_mode()
def similarity_score(x, y):
    """
    Compute similarity score between two tensors.

    Calculates cosine similarity between tensors x and y, handling different
    tensor shapes appropriately.

    Parameters
    ----------
    x : torch.Tensor
        First tensor for similarity computation.
    y : torch.Tensor
        Second tensor for similarity computation.

    Returns
    -------
    torch.Tensor
        Similarity scores. Shape depends on input dimensions.
        For matrices: (x_n, y_n) where x_n and y_n are the number of vectors.
        For vectors: scalar similarity score.

    Raises
    ------
    ValueError
        If tensor shapes are incompatible for similarity computation.

    Examples
    --------
    >>> x = torch.randn(5, 512)
    >>> y = torch.randn(3, 512)
    >>> sim = similarity_score(x, y)
    >>> sim.shape
    torch.Size([5, 3])
    """
    if x.shape != y.shape:
        x_ = torch.nn.functional.normalize(x, dim=-1)
        y_ = torch.nn.functional.normalize(y, dim=-1)
        if x.shape[1] == y.shape[0]:
            return x_.matmul(y_)
        elif x.shape[1] == y.shape[1]:
            return x_.matmul(y_.T)
        raise ValueError("x and y must have the same shape")
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return cos_sim


@torch.inference_mode()
def polysemanticity_score(V, replace_empty_clusters=True, random_state=123, n_clusters=2):
    """
    Compute polysemanticity score for concept representations.

    Measures how polysemantic (multi-meaning) concepts are by clustering
    concept examples and computing clarity of cluster centers. Higher values
    indicate more polysemantic concepts.

    Parameters
    ----------
    V : torch.Tensor
        Concept tensor of shape (n_neurons, n_samples, n_features).
    replace_empty_clusters : bool, default=True
        Whether to replace empty clusters with alternative computation.
    random_state : int, default=123
        Random seed for K-means clustering reproducibility.
    n_clusters : int, default=2
        Number of clusters for K-means algorithm.

    Returns
    -------
    torch.Tensor
        Polysemanticity scores of shape (n_neurons,). Values in range [0, 1],
        where higher values indicate more polysemantic concepts.

    Examples
    --------
    >>> V = torch.randn(10, 20, 512)  # 10 neurons, 20 samples, 512 features
    >>> poly = polysemanticity_score(V)
    >>> poly.shape
    torch.Size([10])
    """
    # V.shape = (n_neurons) x n_samples x n_features
    device = V.device

    clusters = [KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(e.detach().cpu()) for e in V]
    c_centers = torch.stack([torch.from_numpy(c.cluster_centers_) for c in clusters], dim=0).to(device)

    clarity = clarity_score(c_centers)
    poly = 1 - clarity

    if replace_empty_clusters:
        logger.debug("replacing empty cluster")
        # retrieve the ones where a cluster has no samples
        counts = [torch.from_numpy(np.unique(c.labels_, return_counts=True)[1]) for c in clusters]
        counts = torch.stack([x if len(x) == n_clusters else torch.zeros(n_clusters) for x in counts], dim=0)
        v_not = V[counts.amin(-1) < 2]
        if v_not.shape[0] > 0:
            clarity_not = 0
            num_samples = min(10, v_not.shape[1])
            for i in range(num_samples):
                clarity_not += clarity_score(torch.stack([v_not.mean(1), v_not[:, i]], dim=1))
            poly[counts.amin(-1) < 2] = 1 - clarity_not.double() / num_samples
    return poly
