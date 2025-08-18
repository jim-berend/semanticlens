"""
Base classes for foundation models and processors.

This module defines abstract base classes for vision-language foundation models
and their processors, providing a consistent interface for different model
implementations.
"""

from abc import ABC, abstractmethod


class AbstractVLM(ABC):
    """
    Abstract base class for vision-language foundation models.

    This class defines the interface that all vision-language foundation models
    must implement, providing methods for encoding both vision and text inputs.
    """

    @abstractmethod
    def encode_image(self, *args, **kwargs):
        """
        Encode image input into feature representation.

        Parameters
        ----------
        *args
            Variable length argument list for image inputs.
        **kwargs
            Arbitrary keyword arguments for encoding options.

        Returns
        -------
        torch.Tensor
            Encoded image features.
        """
        pass

    @abstractmethod
    def encode_text(self, *args, **kwargs):
        """
        Encode text input into feature representation.

        Parameters
        ----------
        *args
            Variable length argument list for text inputs.
        **kwargs
            Arbitrary keyword arguments for encoding options.

        Returns
        -------
        torch.Tensor
            Encoded text features.
        """
        pass

    @abstractmethod
    def preprocess(self, img):
        """
        Preprocess image input for model consumption.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor to preprocess.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor ready for model input.
        """
        pass

    @abstractmethod
    def tokenize(self, txt: str):
        """
        Tokenize text input for model consumption.

        Parameters
        ----------
        txt : str
            Input text string to tokenize.

        Returns
        -------
        torch.Tensor
            Tokenized text tensor ready for model input.
        """
        pass

    @property
    @abstractmethod
    def device(self):
        """
        Get the device on which the model is located.

        Returns
        -------
        torch.device
            The device (CPU/GPU) on which the model parameters are located.
        """
        pass

    @abstractmethod
    def to(self, device):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The target device to move the model to (e.g., 'cpu', 'cuda:0').

        Returns
        -------
        VisionLanguageFoundationModel
            The model instance after moving to the specified device.
        """
        pass
