# tests/test_scores.py

import pytest
import torch

from semanticlens import scores


# A fixture to provide a sample concept tensor for testing
# Shape: (n_neurons, n_samples, n_features)
@pytest.fixture
def concept_tensor():
    """Provides a sample concept tensor of shape (10, 20, 128)."""
    return torch.randn(10, 20, 128)


# A fixture for an aggregated concept tensor (cones)
# Shape: (n_neurons, n_features)
@pytest.fixture
def aggregated_concept_tensor():
    """Provides a sample aggregated concept tensor of shape (10, 128)."""
    return torch.randn(10, 15, 128)


def test_clarity_score(concept_tensor):
    """
    Tests the clarity_score function.
    """
    clarity = scores.clarity_score(concept_tensor)

    # Check for correct output shape
    assert clarity.shape == (10,)

    # Check that values are within the expected range [-1/(n-1), 1]
    num_samples = concept_tensor.shape[1]
    assert torch.all(clarity >= -1.0 / (num_samples - 1))
    assert torch.all(clarity <= 1.0)


def test_redundancy_score(aggregated_concept_tensor):
    """
    Tests the redundancy_score function.
    """
    redundancy = scores.redundancy_score(aggregated_concept_tensor)

    # Check for correct output shape
    assert redundancy.shape == (10,)

    # Redundancy is a cosine similarity, so it should be between -1 and 1
    assert torch.all(redundancy >= -1.0)
    assert torch.all(redundancy <= 1.0)


def test_polysemanticity_score(concept_tensor):
    """
    Tests the polysemanticity_score function.
    """
    # Using a smaller tensor to speed up KMeans
    small_concept_tensor = concept_tensor[:5, :10]
    poly = scores.polysemanticity_score(small_concept_tensor)

    # Check for correct output shape
    assert poly.shape == (5,)

    assert torch.all(poly >= 0.0)
