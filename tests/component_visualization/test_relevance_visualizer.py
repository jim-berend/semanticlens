import json
import tempfile

import pytest
import torch
from crp.attribution import CondAttribution as Attribution
from torch.nn import Linear, Sequential
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from zennit.composites import EpsilonPlusFlat as Composite

from semanticlens.component_visualization import (
    RelevanceComponentVisualizer,
)


@pytest.fixture
def mock_model():
    return Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))


@pytest.fixture
def mock_dataset():
    # Using a dummy dataset path for testing purposes
    return ImageFolder("/data/datapool/datasets/ImageNet-complete/val", transform=T.ToTensor())  # TODO other dummy data


def test_activation_component_visualizer_initialization(mock_model, mock_dataset):
    model = mock_model
    attribution = Attribution(model)
    composite = Composite()
    with tempfile.TemporaryDirectory() as temp_dir:
        rel_cv = RelevanceComponentVisualizer(
            attribution,
            mock_dataset,
            layer_names=["0", "1", "2"],
            preprocess_fn=T.Normalize((0.5,), (0.5,)),
            aggregation_fn="sum",
            composite=composite,
            abs_norm=True,
            storage_dir=temp_dir,
            device=torch.device("cuda"),
            cache=None,
            num_samples=2,
        )
        assert rel_cv is not None
        assert json.dumps(rel_cv.metadata, sort_keys=True) is not None
