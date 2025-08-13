"""
Base classes for foundation models and processors.

This module defines abstract base classes for vision-language foundation models
and their processors, providing a consistent interface for different model
implementations.
"""

import abc

from torch.nn import Module


class VisionLanguageFoundationModel(Module, abc.ABC):
    """
    Abstract base class for vision-language foundation models.

    This class defines the interface that all vision-language foundation models
    must implement, providing methods for encoding both vision and text inputs.

    Methods
    -------
    encode_vision(*args, **kwargs)
        Encode visual inputs into embeddings.
    encode_text(*args, **kwargs)
        Encode text inputs into embeddings.
    """

    @abc.abstractmethod
    def encode_vision(self, *args, **kwargs):
        """
        Encode visual inputs into embeddings.

        Parameters
        ----------
        *args
            Variable length argument list for visual inputs.
        **kwargs
            Arbitrary keyword arguments for visual encoding.

        Returns
        -------
        torch.Tensor
            Visual embeddings.
        """
        ...

    @abc.abstractmethod
    def encode_text(self, *args, **kwargs):
        """
        Encode text inputs into embeddings.

        Parameters
        ----------
        *args
            Variable length argument list for text inputs.
        **kwargs
            Arbitrary keyword arguments for text encoding.

        Returns
        -------
        torch.Tensor
            Text embeddings.
        """
        ...


class VisionLanguageProcessor(abc.ABC):
    """
    Abstract base class for vision-language processors.

    This class defines the interface for preprocessing vision and text inputs
    for foundation models.

    Methods
    -------
    __call__(*, images, text, **kwargs)
        Process images and/or text inputs.
    """

    @abc.abstractmethod
    def __call__(self, *, images, text, **kwargs):
        """
        Process images and/or text inputs for model consumption.

        Parameters
        ----------
        images : optional
            Image data to be processed.
        text : optional
            Text data to be processed.
        **kwargs
            Additional processing arguments.

        Returns
        -------
        dict
            Processed inputs ready for model consumption.
        """
        ...
