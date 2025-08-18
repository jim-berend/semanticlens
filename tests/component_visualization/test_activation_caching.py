# tests/component_visualization/test_activation_caching.py

import pytest
import torch
import torch.nn as nn

from semanticlens.component_visualization.activation_caching import ActMax, ActMaxCache
from semanticlens.component_visualization.aggregators import aggregate_conv_mean


class TestActMax:
    """Tests the ActMax class for collecting top-k activations."""

    def test_initialization_and_update(self):
        act_max = ActMax(n_collect=5, n_latents=3)
        assert act_max.is_setup

        # First update
        acts1 = torch.tensor([[0.1, 0.9, 0.3], [0.2, 0.8, 0.4]])
        ids1 = torch.tensor([0, 1])
        act_max.update(acts1, ids1)

        # Second update
        acts2 = torch.tensor([[0.9, 0.1, 0.5], [0.8, 0.2, 0.6]])
        ids2 = torch.tensor([2, 3])
        act_max.update(acts2, ids2)

        # Check final activations (should be sorted descending)
        assert torch.allclose(act_max.activations[0], torch.tensor([0.9, 0.8, 0.2, 0.1, 0.0]).to(torch.bfloat16))
        assert torch.allclose(act_max.sample_ids[0], torch.tensor([2, 3, 1, 0, -1]))

    def test_store_and_load(self, tmp_path):
        """Ensures that saving and loading an ActMax instance works correctly."""
        file_path = tmp_path / "actmax.safetensors"
        act_max_original = ActMax(n_collect=5, n_latents=3)
        acts = torch.rand(10, 3)
        ids = torch.arange(10)
        act_max_original.update(acts, ids)

        metadata = {"n_collect": "5", "n_latents": "3"}
        act_max_original.store(file_path, metadata=metadata)

        act_max_loaded = ActMax.load(file_path)

        assert act_max_loaded.n_collect == 5
        assert act_max_loaded.n_latents == 3
        assert torch.allclose(act_max_original.activations, act_max_loaded.activations)
        assert torch.all(act_max_original.sample_ids == act_max_loaded.sample_ids)


class TestActMaxCache:
    """Tests the ActMaxCache for hooking into a model and managing ActMax instances."""

    @pytest.fixture
    def simple_model(self):
        """A simple nn.Module for testing hooks."""
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 16, 3))
        model.layer_names = ["0", "2"]
        return model

    def test_hook_context(self, simple_model):
        """Tests that the hook context correctly captures activations."""
        cache = ActMaxCache(layer_names=simple_model.layer_names, aggregation_fn=aggregate_conv_mean, n_collect=10)

        with cache.hook_context(simple_model):
            # Run a dummy forward pass
            simple_model(torch.randn(4, 3, 32, 32))

        # Check if the cache was populated
        assert cache.cache["0"].is_setup
        assert cache.cache["2"].is_setup
        assert cache.cache["0"].activations.shape == (8, 10)  # 8 channels, 10 samples
        assert cache.cache["2"].activations.shape == (16, 10)  # 16 channels, 10 samples

    def test_store_and_load(self, simple_model, tmp_path):
        """Ensures that the entire ActMaxCache can be saved and reloaded."""
        storage_dir = tmp_path / "actmax_cache"
        layer_names = ["0", "2"]

        # Populate and store the cache
        cache_original = ActMaxCache(layer_names=layer_names, aggregation_fn=aggregate_conv_mean, n_collect=10)
        with cache_original.hook_context(simple_model):
            simple_model(torch.randn(4, 3, 32, 32))
        cache_original.store(storage_dir)

        # Create a new instance and load from disk
        cache_loaded = ActMaxCache(layer_names=layer_names, aggregation_fn=aggregate_conv_mean, n_collect=10)
        cache_loaded.load(storage_dir)

        assert cache_loaded.cache.keys() == cache_original.cache.keys()
        assert torch.allclose(cache_original.cache["0"].activations, cache_loaded.cache["0"].activations)
