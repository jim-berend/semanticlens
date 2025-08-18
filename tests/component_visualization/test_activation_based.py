import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from semanticlens.component_visualization.activation_based import ActivationComponentVisualizer, MissingNameWarning


@pytest.fixture
def mock_model():
    """Provides a mock model with named modules."""
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 16, 3))
    model.layer_names = ["0", "2"]
    model.name = "mock_model"
    return model


@pytest.fixture
def mock_dataset():
    """Provides a mock dataset."""
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32))
    dataset.name = "mock_dataset"
    return dataset


def test_initialization(mock_model, mock_dataset):
    """Tests the successful initialization of the visualizer."""
    layer_names = ["0"]
    visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset,
        dataset_fm=mock_dataset,
        layer_names=layer_names,
        num_samples=10,
        cache_dir=None,
    )
    assert visualizer.model is mock_model
    assert visualizer.layer_names == layer_names
    assert not visualizer.caching


def test_initialization_fails_with_bad_layer(mock_model, mock_dataset):
    """Tests that initialization fails if a layer name is not found."""
    with pytest.raises(ValueError, match="Layer 'bad_layer' not found in model"):
        ActivationComponentVisualizer(
            model=mock_model,
            dataset_model=mock_dataset,
            dataset_fm=mock_dataset,
            layer_names=["bad_layer"],
            num_samples=10,
        )


def test_missing_name_warning(mock_model, mock_dataset, tmp_path):
    """Tests that a warning is raised if model or dataset lacks a .name attribute when caching."""
    # Remove the .name attribute for the test
    del mock_model.name

    with pytest.warns(MissingNameWarning, match="Model does not have a name attribute"):
        ActivationComponentVisualizer(
            model=mock_model,
            dataset_model=mock_dataset,
            dataset_fm=mock_dataset,
            layer_names=["0"],
            num_samples=10,
            cache_dir=str(tmp_path),
        )


def test_run_loads_from_cache_if_available(mock_model, mock_dataset, tmp_path, mocker):
    """Tests that the `run` method loads from cache and does not re-compute if a cache file is found."""
    # Mock the load method to signal a cache hit
    mock_load = mocker.patch(
        "semanticlens.component_visualization.activation_caching.ActMaxCache.load", return_value={}
    )
    # Mock the internal _run method to ensure it's NOT called
    mock_run_computation = mocker.patch(
        "semanticlens.component_visualization.activation_based.ActivationComponentVisualizer._run"
    )

    visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset,
        dataset_fm=mock_dataset,
        layer_names=["0"],
        num_samples=10,
        cache_dir=str(tmp_path),
    )
    visualizer.run()

    mock_load.assert_called()
    assert mock_load.call_count == 2
    mock_run_computation.assert_not_called()


def test_run_triggers_computation_and_stores_on_cache_miss(mock_model, mock_dataset, tmp_path, mocker):
    """
    Tests that the public `run` method triggers a computation and stores
    the result when the cache is not found. This correctly tests the public behavior
    without relying on private implementation details.
    """
    # Mock the load method to simulate a cache miss
    mocker.patch(
        "semanticlens.component_visualization.activation_caching.ActMaxCache.load",
        side_effect=FileNotFoundError,
    )
    # Mock the store method to verify it gets called
    mock_store = mocker.patch("semanticlens.component_visualization.activation_caching.ActMaxCache.store")

    visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset,
        dataset_fm=mock_dataset,
        layer_names=["0"],
        num_samples=10,
        cache_dir=str(tmp_path),
    )

    # Call the public run method
    visualizer.run(batch_size=2)

    # Assert that store was called, which implies the computation ran
    mock_store.assert_called_once()


def test_initialization_with_empty_layer_names(mock_model, mock_dataset):
    """Tests that initialization is successful with an empty list of layer names and run completes."""
    visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset,
        dataset_fm=mock_dataset,
        layer_names=[],
        num_samples=10,
        cache_dir=None,
    )
    assert visualizer.layer_names == []
    # The run should complete without errors and return an empty result
    result = visualizer.run()
    assert result == {}


def test_run_with_zero_samples(mock_model, mock_dataset, tmp_path):
    """Tests behavior when num_samples is zero, which should result in empty caches."""
    visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset,
        dataset_fm=mock_dataset,
        layer_names=["0"],
        num_samples=0,
        cache_dir=str(tmp_path),
    )

    # A run should work and the cache should reflect 0 samples collected.
    visualizer.run(batch_size=2)

    assert "0" in visualizer.actmax_cache.cache
    actmax_instance = visualizer.actmax_cache.cache["0"]
    assert actmax_instance.n_collect == 0
    # Tensors should be initialized but have a size of 0 in the collection dimension
    assert actmax_instance.sample_ids.shape[1] == 0
    assert actmax_instance.activations.shape[1] == 0
