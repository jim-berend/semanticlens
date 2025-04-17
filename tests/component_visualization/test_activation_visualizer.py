import json
import tempfile

import pytest
from torch.nn import Linear, Sequential
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from semanticlens.component_visualization import ActivationComponentVisualizer


@pytest.fixture
def mock_model():
    return Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))


@pytest.fixture
def mock_dataset():
    # Using a dummy dataset path for testing purposes
    return ImageFolder("/data/datapool/datasets/ImageNet-complete/val", transform=T.ToTensor())


def test_activation_component_visualizer_initialization(mock_model, mock_dataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        act_cv = ActivationComponentVisualizer(
            model=mock_model,
            dataset=mock_dataset,
            layer_names=["0", "1", "2"],
            aggregation_fn="max",
            num_samples=100,
            storage_dir=temp_dir,
        )
        assert act_cv is not None
        assert json.dumps(act_cv.metadata, sort_keys=True) is not None
