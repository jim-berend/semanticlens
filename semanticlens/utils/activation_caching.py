"""Helpers to collect, aggregate and cache activations in torch.

This module provides classes and utilities for collecting, aggregating, and caching
neural network activations during inference. It includes tools for:

- Collecting maximal activations (ActMax, LazyActMax)
- Caching activations during model forward passes (ActCache, AggregationCache, ActMaxCache)
- Computing online statistics using Welford's algorithm (WelfordAlgorithm)

Classes
-------
ActMax
    Tool for collecting and storing maximal scalar activations.
LazyActMax
    Lazy version of ActMax with parameters inferred from first call.
ActCache
    Collect activations from a model during inference.
AggregationCache
    Collect aggregated activations from a model during inference.
ActMaxCache
    Combines ActMax-Sampling with CollectCache for memory-efficient collection.
WelfordAlgorithm
    Online algorithm for computing mean, variance and standard deviation.

Examples
--------
>>> model = torchvision.models.resnet50(pretrained=True)
>>> act_cache = ActCache(["layer1", "layer2"])
>>> with act_cache.hook_context(model):
...     for batch in dataloader:
...         out = model(batch[0].cuda())
>>> layer1_activations = act_cache.cache["layer1"]
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat, pprint

import safetensors.torch
import torch

logger = logging.getLogger(__name__)

# TODO reduce redundancy


class ActMax:
    """Tool for collecting and storing maximal scalar activations.

    Collects and stores the top-k activations for neural network components
    such as neuron activations (MLP or pooled Conv-Filters/Transformer tokens).

    NOTE: This is not (yet) optimized for performance.

    Parameters
    ----------
    n_collect : int
        Number of top activations to collect per latent.
    n_latents : int
        Number of latent dimensions/neurons.
    batch_size : int, optional
        Batch size for processing. If None, will be inferred from first update.

    Attributes
    ----------
    activations : torch.Tensor
        Tensor storing the top activations of shape (n_latents, n_collect).
    sample_ids : torch.Tensor
        Tensor storing sample IDs corresponding to activations.
    welford : WelfordAlgorithm, optional
        Online statistics calculator for mean and variance.
    """

    def __init__(
        self,
        n_collect: int,
        n_latents: int,
        batch_size: int | None = None,
    ):
        self.n_collect = n_collect
        self.n_latents = n_latents
        self.activations = -torch.zeros(n_latents, n_collect, dtype=torch.bfloat16)
        self.sample_ids = -torch.ones(n_latents, n_collect, dtype=torch.int64)
        # self.latent_ids = -torch.ones( n_latents,n_collect, dtype=torch.int64)
        # ╰─ NOTE latent Ids are not needed as we store the full matrix
        self.welford = None

        self.batch_size = batch_size
        if batch_size:
            self._init_batch_acts_tensor()

    def reset(self):
        """Resets the activations and sample IDs to their initial state."""
        self.activations = -torch.zeros(self.n_latents, self.n_collect, dtype=torch.bfloat16)
        self.sample_ids = -torch.ones(self.n_latents, self.n_collect, dtype=torch.int64)
        self.welford = None
        self.batch_size = None
        if hasattr(self, "batch_acts"):
            del self.batch_acts
            del self.batch_ids

    def _init_batch_acts_tensor(self):
        self.batch_acts = (-1) * torch.ones(self.n_latents, self.batch_size, dtype=torch.bfloat16)
        self.batch_ids = (-1) * torch.ones(self.n_latents, self.batch_size, dtype=torch.int64)

    def __call__(self, acts, latent_ids, sample_ids):
        """Update activations using callable interface.

        Parameters
        ----------
        acts : torch.Tensor
            Activation tensor to update with.
        latent_ids : torch.Tensor
            Latent dimension indices.
        sample_ids : torch.Tensor
            Sample indices corresponding to activations.
        """
        self.update(acts, latent_ids, sample_ids)

    def __repr__(self):
        """Return string representation of ActMax instance.

        Returns
        -------
        str
            String representation including dimensions and active latents count.
        """
        return (
            f"ActMax(n_latents={self.n_latents}, n_collect={self.n_collect}, batch_size={self.batch_acts.shape[0]})"
            + f"[n_active={len(self.alive_latents)}]"
        )

    def update(self, acts, sample_ids, latent_ids=None):
        """Updates the batch activations and sample IDs with the given activations and sample IDs.

        Supports full latent code or only active latents + latent ids

        Parameters
        ----------
            acts [batch_size x n_active/n_latents](torch.Tensor): The activations to update with.
            sample_ids (torch.Tensor): The sample IDs corresponding to the activations.
            latent_ids (torch.Tensor, optional): The latent IDs for scattering the activations. Defaults to None.

        Returns
        -------
            None

        """
        if self.batch_size is None:
            self.batch_size = acts.shape[0]
            self._init_batch_acts_tensor()

        if acts.dtype != self.batch_acts.dtype:
            # TODO maybe change the type of the incoming instead!
            self.batch_acts = self.batch_acts.to(acts.dtype)
            self.activations = self.activations.to(acts.dtype)

        # scatter # TODO add counter for active overall and mean/std activations
        self.batch_acts.zero_()
        if latent_ids is not None:
            self.batch_acts.scatter_(dim=0, index=latent_ids.T, src=acts.T)
        else:
            assert acts.shape[1] == self.n_latents, (
                f"If latent_ids==None full latent-code is expected {self.n_latents} got {acts.shape[1]}"
            )
            self.batch_acts[:, : acts.shape[0]] = acts.T

        if self.welford is None:
            self.welford = WelfordAlgorithm(dim=-1, data_shape=self.batch_acts.shape[0])
        self.welford.update(self.batch_acts)

        batch_sample_ids = torch.ones(1, self.batch_acts.shape[1])
        batch_sample_ids[0, : len(sample_ids)] = sample_ids  # ensure batch size
        batch_sample_ids = batch_sample_ids.repeat(self.n_latents, 1).to(self.sample_ids)

        self.activations, ids = torch.cat([self.activations, self.batch_acts], dim=1).topk(k=self.n_collect, dim=1)
        self.sample_ids = torch.cat([self.sample_ids, batch_sample_ids], dim=1).gather(dim=1, index=ids)

    @property
    def alive_latents_mask(self):
        """Get boolean mask for latents with non-zero activations.

        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which latents have any non-zero activations.
        """
        return self.activations.abs().sum(1) > 0

    @property
    def alive_latents(self):
        """Get indices of latents with non-zero activations.

        Returns
        -------
        torch.Tensor
            1D tensor containing indices of latents with non-zero activations.
        """
        return self.alive_latents_mask.nonzero().flatten()

    def store(self, file_path):
        """Store activations and metadata to disk using safetensors format.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to save the data. Can be a directory or specific file path.
            If directory, saves as 'act_max.safetensors'.
            If file without .safetensors extension, creates directory and saves there.
        """
        # TODO support full store and load!
        file_path = Path(file_path)
        if file_path.is_dir():
            file_path = file_path / "act_max.safetensors"
        elif not file_path.suffix == ".safetensors":
            file_path.mkdir(parents=True, exist_ok=True)
            file_path = file_path / "act_max.safetensors"

        alive_latents = self.alive_latents
        tensors = {
            "activations": self.activations[alive_latents, :],
            "sample_ids": self.sample_ids[alive_latents, :],
            "alive_latents": alive_latents,
            "n_collect": torch.tensor(self.n_collect),
            "n_latents": torch.tensor(self.n_latents),
            "batch_size": torch.tensor(self.batch_acts.shape[0]),
            **{k: torch.tensor(v) for k, v in self.welford.get_results().items()},
        }
        safetensors.torch.save_file(tensors, file_path)
        logger.debug("Stored act max samples to %s", file_path)

    @classmethod
    def load(cls, file_path, batch_size: int | None = None):
        """Load activations and metadata from disk using safetensors format.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to load the data from. Can be a directory or specific file path.
            If directory, loads from 'act_max.safetensors'.
        batch_size : int, optional
            Batch size for the loaded instance. If None, uses saved batch size.

        Returns
        -------
        ActMax
            New ActMax instance with loaded data.
        """
        file_path = Path(file_path)
        if file_path.is_dir():
            file_path = file_path / "act_max.safetensors"

        tensors = safetensors.torch.load_file(file_path)
        activations = tensors["activations"]
        sample_ids = tensors["sample_ids"]
        n_collect = tensors["n_collect"].item()
        n_latents = tensors["n_latents"].item()
        alive_latents = tensors["alive_latents"]
        batch_size = tensors["batch_size"].item() if batch_size is None else batch_size
        for key in ["mean", "sample_std", "sample_var", "var", "std", "num_samples"]:
            welford_results = {}
            if key in tensors:
                welford_results[key] = tensors[key]

        act_max = cls(n_latents=n_latents, n_collect=n_collect, batch_size=sample_ids.shape[0])

        act_max.activations[alive_latents, :] = activations.to(act_max.activations.dtype)
        act_max.sample_ids[alive_latents, :] = sample_ids.to(act_max.sample_ids).to(act_max.sample_ids.dtype)
        act_max.welford_results = welford_results

        return act_max


class LazyActMax(ActMax):
    """Lazy version of ActMax with parameters inferred from first call.

    Less needs to be specified from the start as dimensions are inferred
    from the first call/update.

    Parameters
    ----------
    n_collect : int
        Number of top activations to collect per latent.
    n_latents : int, optional
        Number of latent dimensions. If None, inferred from first update.
    batch_size : int, optional
        Batch size for processing. If None, inferred from first update.

    Attributes
    ----------
    index : int
        Counter for tracking batch index in lazy mode.
    """

    def __init__(self, n_collect: int, n_latents: int | None = None, batch_size: int | None = None):
        self.n_collect = n_collect
        self.n_latents = n_latents
        self.batch_size = batch_size
        self.index = 0

        if self.batch_size and self.n_latents:
            self.setup()

    def setup(self):
        """Initialize tensors after n_latents and batch_size are determined.

        Creates the activations, sample_ids, welford, and batch tensors
        with the correct dimensions.
        """
        self.activations = -torch.ones(self.n_latents, self.n_collect, dtype=torch.bfloat16)
        self.sample_ids = -torch.ones(self.n_latents, self.n_collect, dtype=torch.int64)
        self.welford = WelfordAlgorithm(dim=-1, data_shape=self.n_latents)
        self._init_batch_acts_tensor()

    def update(self, acts, sample_ids=None, latent_ids=None):
        """Update activations with lazy initialization of dimensions.

        Parameters
        ----------
        acts : torch.Tensor
            Activation tensor with shape (batch_size, n_latents) or (batch_size, n_active).
        sample_ids : torch.Tensor, optional
            Sample indices. If None, generates sequential indices.
        latent_ids : torch.Tensor, optional
            Latent dimension indices for sparse updates.
        """
        if self.n_latents is None:
            self.batch_size = acts.shape[0]
            self.n_latents = acts.shape[1]
            self.setup()

        if sample_ids is None:
            sample_ids = torch.arange(acts.shape[0]).to(acts) + self.index * self.batch_size
        super().update(acts, sample_ids, latent_ids)
        self.index += 1

    def __repr__(self):
        """Return string representation of LazyActMax instance.

        Returns
        -------
        str
            String representation including setup status and active latents count.
        """
        return f"LazyActMax(n_latents={self.n_latents}, n_collect={self.n_collect}, batch_size={self.batch_size})" + (
            "[not setup yet]" if not self.batch_size or not self.n_latents else f"[n_active={len(self.alive_latents)}]"
        )


class ActCache:
    """Collect activations from a model during inference.

    NOTE this is not optimized and memory constrains may be hit.

    Example:
    -------
    >>> model = torchvision.models.resnet50(pretrained=True)
    >>> act_cache = ActCache(["layer1", "layer2"])
    >>> with act_cache.hook_context(model):
    ...     for batch in dataloader:
    ...         out = model(batch[0].cuda())
    >>>         # ...
    >>> # retrieve activations
    >>> layer1_activations = act_cache.cache["layer1"]
    >>> layer2_activations = act_cache.cache["layer2"]

    """

    def __init__(self, layer_names):
        self.layer_names = layer_names
        self.cache = OrderedDict()
        self.handles = []

    def register_hooks(self, model):
        """Register forward hooks on specified model layers.

        Parameters
        ----------
        model : torch.nn.Module
            The model to register hooks on.
        """
        for name, module in model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self.get_hook(name))
                self.handles.append(handle)

    def get_hook(self, name):
        """Create a forward hook function for a specific layer.

        Parameters
        ----------
        name : str
            Name of the layer to create hook for.

        Returns
        -------
        callable
            Hook function that stores activations in cache.
        """

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.cache[name] = output

        return hook_fn

    @contextmanager
    def hook_context(act_cache, model):
        """Context manager for activation collection.

        Parameters
        ----------
        act_cache : ActCache
            The activation cache instance.
        model : torch.nn.Module
            The model to collect activations from.

        Yields
        ------
        None
            Yields control while hooks are active.
        """
        act_cache.register_hooks(model)
        try:
            yield
        finally:
            for handle in act_cache.handles:
                handle.remove()
            act_cache.handles.clear()
            act_cache._finalize()

    def _finalize(self):
        """A hook for post-processing after the context exits. Can be overridden by subclasses."""
        pass


class AggregationCache(ActCache):
    """Collect aggregated activations from a model during inference.

    Minor extension of ActCache to allow for aggregation of activations.
    See ActCache for more details.


    Custom Aggregation
    ------------------
    >>> def custom_aggregate_fn(cls, tensor):
    ...     # custom aggregation logic
    ...     return tensor.clone().flatten(2).pow(0.5).mean(-1).detach().cpu()
    >>> act_cache = CollectCache(["layer1", "layer2"], aggregate_fn=custom_aggregate_fn)
    >>> with act_cache.hook_context(model):
    ...     for batch in dataloader:
    ...         out = model(batch[0].cuda())
    >>>         # ...
    >>> # retrieve activations
    >>> layer1_activations = act_cache.cache["layer1"]
    >>> layer2_activations = act_cache.cache["layer2"]
    """

    def __init__(self, layer_names, aggregate_fn="max"):
        super().__init__(layer_names)
        if aggregate_fn == "max":
            # logger.info("Using max aggregation")
            logger.debug("Using max aggregation")
            self._aggregate_fn = self.max_aggregate_fn
        elif aggregate_fn == "mean":
            # logger.info("Using mean aggregation")
            logger.debug("Using mean aggregation")
            self._aggregate_fn = self.mean_aggregate_fn
        elif isinstance(aggregate_fn, Callable):
            # logger.info("Using custom aggregation")
            logger.debug("Using custom aggregation")
            self._aggregate_fn = aggregate_fn.__get__(self)
        else:
            raise ValueError(f"Invalid aggregate_fn: {aggregate_fn}")

    def aggregate_fn(self, tensor):
        """Overwrite this function to adapt to your architecture.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to aggregate.

        Returns
        -------
        torch.Tensor
            Aggregated tensor.
        """
        return self._aggregate_fn(tensor)

    def max_aggregate_fn(self, tensor):
        """Aggregate tensor by taking maximum along spatial dimensions.

        Parameters
        ----------
        tensor : torch.Tensor or tuple
            Input tensor to aggregate. If tuple, uses first element.

        Returns
        -------
        torch.Tensor
            Aggregated tensor with max pooling applied.
        """
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        return tensor.clone().flatten(2).amax(-1).detach().cpu()

    def mean_aggregate_fn(self, tensor):
        """Aggregate tensor by taking mean along spatial dimensions.

        Parameters
        ----------
        tensor : torch.Tensor or tuple
            Input tensor to aggregate. If tuple, uses first element.

        Returns
        -------
        torch.Tensor
            Aggregated tensor with mean pooling applied.
        """
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        return tensor.clone().flatten(2).mean(-1).detach().cpu()

    def get_hook(self, name):
        """Create a forward hook function that aggregates and stores activations.

        Parameters
        ----------
        name : str
            Name of the layer to create hook for.

        Returns
        -------
        callable
            Hook function that aggregates and stores activations.
        """

        def hook_fn(module, input, output):
            self.cache.setdefault(name, [])
            self.cache[name].append(self.aggregate_fn(output))

        return hook_fn

    def _finalize(self):
        """Concatenate collected tensors after processing."""
        for name, tensors in self.cache.items():
            if not isinstance(tensors, list):
                continue
            self.cache[name] = torch.cat(tensors).cpu()
            logger.debug(f"Layer {name} collected {self.cache[name].shape} tensors")


class ActMaxCache(AggregationCache):
    """Combines ActMax-Sampling with CollectCache - i.e., the aggregated activations are feed into the ActMax instance
    and only the maximal activations are kept. This is useful for large models with many activations.

    See ActCache, CollectCache and ActMax for more details.

    Example:
    _______
    >>> model = torchvision.models.resnet50(pretrained=True)
    >>> act_cache = ActMaxCache(["layer1", "layer2"], aggregate_fn="max",n_collect=100)
    >>> with act_cache.hook_context(model):
    ...     for batch in dataloader:
    ...         out = model(batch[0].cuda())
    >>>         # ...
    >>> # retrieve activations
    >>> layer1_activations = act_cache.cache["layer1"].sample_ids
    >>> layer2_activations = act_cache.cache["layer2"].sample_ids


    """

    def __init__(self, layer_names, aggregation_fn: str = "max", n_collect=100):
        assert aggregation_fn in ["max", "mean"], (
            "aggregate_fn must be 'max' or 'mean'.\n"
            "For custom aggregation inherit from CollectCache and update the `load` and `store` methods."
        )
        self._aggregation_mode = aggregation_fn
        self._n_collect = n_collect
        super().__init__(layer_names, aggregation_fn)
        self.cache = {name: LazyActMax(n_collect=n_collect) for name in layer_names}

    def __repr__(self):
        """Custom representation to include metadata.

        Returns
        -------
        str
            String representation showing cache contents.
        """
        return f"ActMaxCache({pformat(self.cache)})"

    def get_hook(self, name):
        """Create a forward hook function that feeds aggregated activations to ActMax.

        Parameters
        ----------
        name : str
            Name of the layer to create hook for.

        Returns
        -------
        callable
            Hook function that updates ActMax instance with aggregated activations.
        """

        def hook_fn(module, input, output):
            self.cache[name].update(self.aggregate_fn(output))

        return hook_fn

    def store(self, storage_dir: Path):
        """Store the activations to disk using safetensors.

        Parameters
        ----------
        storage_dir : pathlib.Path
            Directory to store the activation files.
        """
        storage_dir.mkdir(parents=True, exist_ok=True)
        for name, cache in self.cache.items():
            cache.store(storage_dir / f"{self._aggregation_mode}-{name}-{self._n_collect}.safetensors")
        logger.info("Stored act max samples to %s", storage_dir)

    @classmethod
    def load(cls, storage_dir: Path, aggregation_fn: str = "max"):
        """Load the activations from disk using safetensors.

        Parameters
        ----------
        storage_dir : pathlib.Path
            Directory containing the stored activation files.
        aggregation_fn : str, default "max"
            Aggregation function used when storing ("max" or "mean").

        Returns
        -------
        ActMaxCache
            New ActMaxCache instance with loaded data.

        Raises
        ------
        FileNotFoundError
            If storage directory doesn't exist or no matching files found.
        """
        storage_dir = Path(storage_dir) if isinstance(storage_dir, str) else storage_dir
        if not storage_dir.is_dir():
            raise FileNotFoundError(f"Storage directory {storage_dir} does not exist.")

        fpaths = [fpath for fpath in storage_dir.glob("*.safetensors") if fpath.stem.startswith(aggregation_fn)]
        if not fpaths:
            raise FileNotFoundError(f"No safetensors found for {aggregation_fn} in {storage_dir}")

        n_collect = int(fpaths[0].stem.split("-")[2])
        layer_names = [fpath.stem.split("-")[1] for fpath in fpaths]

        if hasattr(cls, "cache"):
            # instance is already initialized -> check that the parameters match
            if cls._aggregation_mode != aggregation_fn or cls._n_collect != n_collect or cls.layer_names != layer_names:
                raise FileNotFoundError(
                    f"Aggregation function {aggregation_fn} does not match the stored aggregation function {cls._aggregation_mode}.\n"
                    + f"n_collect {n_collect} does not match the stored n_collect {cls._n_collect}.\n"
                    + f"Layer names {layer_names} do not match the stored layer names {cls.layer_names}."
                )

        actmax_cache = cls(layer_names, aggregation_fn=aggregation_fn, n_collect=n_collect)

        for fpath, layer_name in zip(fpaths, layer_names):
            actmax_cache.cache[layer_name] = LazyActMax.load(fpath)

        return actmax_cache


class WelfordAlgorithm:
    """Online Algorithm for computing
    - Mean
    - Variance (sample)
    - Standard Derivation (sample)
    handles batches.

    see: [Algorithms_for_calculating_variance@Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version)
    """

    def __init__(self, dim=0, data_shape=(1,), device="cpu"):
        self.n = 0
        self.dim = dim
        self.data_shape = data_shape
        self.mean = torch.zeros(data_shape).to(device).squeeze(self.dim)
        self.M2 = torch.zeros(data_shape).to(device)
        self.device = device

    def update(self, batch_values: torch.Tensor):
        """Update statistics with a new batch of values.

        Parameters
        ----------
        batch_values : torch.Tensor
            Batch of values to incorporate into running statistics.

        Returns
        -------
        WelfordAlgorithm
            Self for method chaining.
        """
        batch_values = batch_values.to(self.device)

        while len(batch_values.shape) <= 1:
            batch_values = batch_values.unsqueeze(-1)  # 1dim is batch dim  2dim is data dim

        # batch_mean = batch_values.mean(dim=self.dim, keepdim=True)
        batch_len = batch_values.shape[self.dim]

        # if self.n == 0 and self.mean.shape != batch_mean.shape:
        #     self.__init__(data_shape=tuple(batch_mean.shape))

        self.n += batch_len
        delta = batch_values.mean(dim=self.dim) - self.mean
        self.mean += delta * batch_len / self.n
        self.M2 += ((batch_values - batch_values.mean(dim=self.dim, keepdim=True)) ** 2).sum(dim=self.dim)
        self.M2 += delta**2 * batch_len * (self.n - batch_len) / self.n

        return self

    def reset(self):
        """Reset the algorithm to initial state.

        Clears all accumulated statistics and reinitializes the algorithm.
        """
        logger.info(self.__dict__)
        logger.info("Reset.")
        self.__init__()

    def get_results(self):
        """Get the computed statistics.

        Returns
        -------
        dict
            Dictionary containing computed statistics with keys:
            - 'mean': Sample mean
            - 'sample_std': Sample standard deviation
            - 'sample_var': Sample variance
            - 'var': Population variance
            - 'std': Population standard deviation
            - 'num_samples': Number of samples processed
        """
        if not self.n > 2:
            logger.warning("Not enough data seen.")
            sample_var = 0
            var = 0
            sample_std = 0
            std = 0
            mean = self.mean
        else:
            sample_var = self.M2 / (self.n - 1)
            var = self.M2 / self.n
            sample_std = sample_var**0.5
            std = var**0.5
            mean = self.mean

        res = {
            "mean": mean,
            "sample_std": sample_std,
            "sample_var": sample_var,
            "var": var,
            "std": std,
            "num_samples": self.n,
        }
        return res

    def __repr__(self):
        """Return string representation of computed statistics.

        Returns
        -------
        str
            Pretty-formatted representation of current statistics.
        """
        res = self.get_results()
        return pprint.pformat(res)
