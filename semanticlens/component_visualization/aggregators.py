"""
Aggregation functions for neural network activations.

This module provides various aggregation functions for reducing tensor dimensions
in neural network activations. These functions are used to summarize activations
across spatial or temporal dimensions for further analysis.

Functions
---------
aggregate_conv_mean : callable
    Aggregates 4D convolutional tensors by taking the mean over spatial dimensions.
aggregate_conv_max : callable
    Aggregates 4D convolutional tensors by taking the max over spatial dimensions.
aggregate_transformer_mean : callable
    Aggregates 3D transformer tensors by taking the mean over token dimension.
aggregate_transformer_absmean : callable
    Aggregates 3D transformer tensors by taking the mean of absolute values over token dimension.
aggregate_transformer_max : callable
    Aggregates 3D transformer tensors by taking the max over token dimension.
aggregate_transformer_absmax : callable
    Aggregates 3D transformer tensors by taking the max of absolute values over token dimension.
get_aggregate_transformer_special_token : callable
    Returns a function that extracts values at a specific token position.

Notes
-----
The function names are involved during caching and should be kept consistent.
"""

import torch

# NOTE the function names are involved during caching!


_ERROR_MESSAGE = f"(Select or implement a different aggregation function in {__file__}.)"


def aggregate_conv_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 4D convolutional tensor by taking the mean over spatial dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 4D tensor with shape (batch, channels, height, width).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, channels) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 4-dimensional.
    """
    if tensor.ndim != 4:
        raise ValueError("Input tensor should be 4D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().flatten(2).mean(-1).detach().cpu()


def aggregate_conv_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 4D convolutional tensor by taking the max over spatial dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 4D tensor with shape (batch, channels, height, width).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, channels) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 4-dimensional.
    """
    if tensor.ndim != 4:
        raise ValueError("Input tensor should be 4D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().flatten(2).amax(-1).detach().cpu()


def aggregate_transformer_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 3D transformer tensor by taking the mean over the token dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor with shape (batch, tokens, features).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, features) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 3-dimensional.
    """
    # Assumes shape (batch, tokens, features)
    if tensor.ndim != 3:
        raise ValueError("Input tensor should be 3D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().mean(1).detach().cpu()


def aggregate_transformer_absmean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 3D transformer tensor by taking the mean of absolute values over the token dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor with shape (batch, tokens, features).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, features) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 3-dimensional.
    """
    # Assumes shape (batch, tokens, features)
    if tensor.ndim != 3:
        raise ValueError("Input tensor should be 3D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().abs().mean(1).detach().cpu()


def aggregate_transformer_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 3D transformer tensor by taking the max over the token dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor with shape (batch, tokens, features).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, features) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 3-dimensional.
    """
    # Assumes shape (batch, tokens, features)
    if tensor.ndim != 3:
        raise ValueError("Input tensor should be 3D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().amax(1).detach().cpu()


def aggregate_transformer_absmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a 3D transformer tensor by taking the max of absolute values over the token dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor with shape (batch, tokens, features).

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (batch, features) on CPU.

    Raises
    ------
    ValueError
        If input tensor is not 3-dimensional.
    """
    # Assumes shape (batch, tokens, features)
    if tensor.ndim != 3:
        raise ValueError("Input tensor should be 3D. \n" + _ERROR_MESSAGE)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.clone().abs().amax(1).detach().cpu()


def get_aggregate_transformer_special_token(token_position: int):
    """
    Return a function that aggregates a 3D tensor by extracting values at a specific token position.

    Parameters
    ----------
    token_position : int
        The position of the token to extract values from.

    Returns
    -------
    callable
        A function that takes a 3D tensor and returns values at the specified token position.

    Examples
    --------
    >>> agg_fn = get_aggregate_transformer_special_token(0)  # Extract CLS token
    >>> result = agg_fn(tensor)  # tensor shape: (batch, tokens, features)
    """

    def aggregate_transformer_special_token(tensor: torch.Tensor) -> torch.Tensor:
        """
        Aggregate a 3D transformer tensor by extracting values at a specific token position.

        Parameters
        ----------
        tensor : torch.Tensor
            Input 3D tensor with shape (batch, tokens, features).

        Returns
        -------
        torch.Tensor
            Tensor with shape (batch, features) containing values at the specified token position.

        Raises
        ------
        ValueError
            If input tensor is not 3-dimensional.
        """
        # Assumes shape (batch, tokens, features)
        if tensor.ndim != 3:
            raise ValueError("Input tensor should be 3D. \n" + _ERROR_MESSAGE)
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        return tensor.clone()[:, token_position].detach().cpu()

    return aggregate_transformer_special_token
