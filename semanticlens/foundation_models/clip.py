"""
CLIP model implementations for vision-language tasks.

This module provides concrete implementations of various CLIP model variants,
including OpenCLIP, SigLIP V2, and MobileCLIP models for encoding both
images and text into a shared embedding space.

# TODO refine

Classes
-------
OpenClip : VisionLanguageFoundationModel
    OpenCLIP implementation supporting various model architectures.
SigLipV2 : OpenClip
    SigLIP V2 model implementation.
ClipMobile : OpenClip
    MobileCLIP model implementation optimized for mobile deployment.
"""

from __future__ import annotations

import torch
from PIL import Image

from semanticlens.foundation_models.base import VisionLanguageFoundationModel


class OpenClip(VisionLanguageFoundationModel):
    """
    OpenCLIP vision-language model implementation.

    This class provides a concrete implementation of the VisionLanguageFoundationModel
    abstract base class using OpenCLIP models. It supports encoding images and text
    into a shared embedding space for various vision-language tasks.

    Parameters
    ----------
    url : str
        The model URL or identifier for OpenCLIP model loading.
    device : str, optional
        The device to load the model on, by default "cpu".

    Attributes
    ----------
    model : torch.nn.Module
        The loaded OpenCLIP model.
    preprocessor : callable
        Image preprocessing function.
    tokenizer : callable
        Text tokenization function.
    """

    def __init__(self, url, device="cpu"):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            url,
        )
        tokenizer = open_clip.get_tokenizer(url)

        self.model = model.eval().to(device)

        self.preprocessor = preprocess
        self.tokenizer = tokenizer

    @property
    def device(self):
        """
        Get the device on which the model is located.

        Returns
        -------
        torch.device
            The device (CPU/GPU) on which the model parameters are located.
        """
        return next(self.model.parameters()).device

    def to(self, device):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The target device to move the model to (e.g., 'cpu', 'cuda:0').

        Returns
        -------
        torch.nn.Module
            The model instance after moving to the specified device.
        """
        return self.model.to(device)

    def encode_image(self, img: torch.Tensor):
        """
        Encode an image tensor into features.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Encoded image features.
        """
        with torch.no_grad():
            return self.model.encode_image(img)

    def encode_text(self, text_input: torch.Tensor):
        """
        Encode a text tensor into features.

        Parameters
        ----------
        text_input : torch.Tensor
            Input text tensor.

        Returns
        -------
        torch.Tensor
            Encoded text features.
        """
        with torch.no_grad():
            return self.model.encode_text(text_input)

    def preprocess(self, img: Image.Image | list[Image.Image]) -> torch.Tensor:
        """
        Apply foundation model image preprocessing.

        Preprocesses images for model consumption, handling both single images
        and lists of images. Also handles tensor dimension expansion and device
        placement automatically.

        Parameters
        ----------
        img : Image.Image or list[Image.Image]
            Input image(s) to preprocess. Can be a single PIL Image or a list
            of PIL Images.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor(s) ready for model input, moved to the
            correct device with proper batch dimensions.
        """
        if isinstance(img, list):
            img_inputs = torch.stack([self.preprocessor(image) for image in img])
        else:
            img_inputs = self.preprocessor(img)
        if img_inputs.ndim == 3:
            img_inputs = img_inputs.unsqueeze(0)
        return img_inputs.to(self.device)

    def tokenize(self, txt: str, context_length=None):
        """
        Tokenize a text string and move to the correct device.

        Converts input text into tokenized format suitable for the model,
        automatically moving the result to the model's device.

        Parameters
        ----------
        txt : str
            Input text string to tokenize.
        context_length : int, optional
            Maximum context length for tokenization. If None, uses the
            model's default context length.

        Returns
        -------
        torch.Tensor
            Tokenized text tensor ready for model input, on the correct device.
        """
        context_length = context_length or self.model.context_length
        text_inputs = self.tokenizer(txt, context_length=context_length)
        return text_inputs.to(self.device)


class SigLipV2(OpenClip):
    """
    SigLIP V2 vision-language model implementation.

    A specialized OpenCLIP implementation using the SigLIP V2 model architecture
    optimized for improved vision-language understanding.

    Parameters
    ----------
    device : str, optional
        The device to load the model on, by default "cpu".

    Attributes
    ----------
    URL : str
        The model identifier for SigLIP V2 model loading.
    """

    URL = "hf-hub:timm/ViT-B-16-SigLIP2"

    def __init__(self, device="cpu"):
        super().__init__(url=self.URL, device=device)


class ClipMobile(OpenClip):
    """
    MobileCLIP vision-language model implementation.

    A specialized OpenCLIP implementation using MobileCLIP models optimized
    for mobile deployment with efficient inference while maintaining performance.

    Parameters
    ----------
    version : str, optional
        The MobileCLIP version to use ('s1' or 's2'), by default "s1".
    device : str, optional
        The device to load the model on, by default "cpu".

    Attributes
    ----------
    URLs : dict
        Dictionary mapping version names to model identifiers.
    """

    URLs = dict(s1="MobileCLIP-S1", s2="MobileCLIP-S2")

    def __init__(self, version="s1", device="cpu"):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(self.URLs[version], pretrained="datacompdr")
        tokenizer = open_clip.get_tokenizer(self.URLs[version])

        self.model = model.eval().to(device)

        self.preprocessor = preprocess
        self.tokenizer = tokenizer

    @staticmethod
    def _test():
        # TODO (re)move!
        def load_dummy_image_via_request():
            # Load a dummy image from the internet
            from io import BytesIO

            import requests
            # from PIL import Image

            url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            return img.convert("RGB")

        model = ClipMobile(use_half=True, device="cpu", version="s1")
        img = load_dummy_image_via_request()
        text = ["This is a test sentence.", "one di", "two dice", "three dice"]

        text_input = model.tokenize(text)
        img_input = model.preprocess(img).unsqueeze(0)  # Add batch dimension

        img_features = model.encode_image(img_input)
        text_features = model.encode_text(text_input)

        scores = img_features @ text_features.T

        print("-" * 80)
        print("Testing CLIP MobileS model")
        print("-" * 80)
        print("texts", text)
        print("Scores shape:", scores)
        if scores.argmax() == text.index("three dice"):
            print("Correctly identified the last text as the best match.")
        else:
            print("Did not identify the last text as the best match.")

        print("Image features shape:", img_features.shape)
        print("Text features shape:", text_features.shape)
