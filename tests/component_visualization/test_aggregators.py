# tests/component_visualization/test_aggregators.py

import pytest
import torch

from semanticlens.component_visualization import aggregators


# Test for convolutional aggregators (expect 4D input)
@pytest.mark.parametrize(
    "agg_func",
    [
        (aggregators.aggregate_conv_mean),
        (aggregators.aggregate_conv_max),
    ],
)
def test_conv_aggregators(
    agg_func,
):
    """
    Tests that convolutional aggregators correctly reduce a 4D tensor
    to a 2D tensor with the expected values.
    """
    tensor_4d = torch.randn(2, 4, 8, 8)  # (batch, channels, H, W)
    result = agg_func(tensor_4d)

    # Check for correct shape
    assert result.shape == (2, 4)

    # Check that it raises an error for incorrect dimensions
    tensor_3d = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="Input tensor should be 4D"):
        agg_func(tensor_3d)


# Test for transformer aggregators (expect 3D input)
@pytest.mark.parametrize(
    "agg_func",
    [(aggregators.aggregate_transformer_mean), (aggregators.aggregate_transformer_max)],
)
def test_transformer_aggregators(agg_func):
    """
    Tests that transformer aggregators correctly reduce a 3D tensor
    to a 2D tensor along the token dimension.
    """
    tensor_3d = torch.randn(2, 10, 16)  # (batch, tokens, features)
    result = agg_func(tensor_3d)

    assert result.shape == (2, 16)

    # Check for error on incorrect dimensions
    tensor_4d = torch.randn(2, 10, 16, 1)
    with pytest.raises(ValueError, match="Input tensor should be 3D"):
        agg_func(tensor_4d)
