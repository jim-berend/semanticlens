"""
CLIP model implementations for vision-language foundation models.

This module provides implementations of CLIP models from different sources:
OpenCLIP and Hugging Face transformers, along with their processors.
"""

import open_clip
import torch
from transformers import CLIPModel, CLIPProcessor

from semanticlens.foundation_models.base import VisionLanguageFoundationModel, VisionLanguageProcessor


class TorchDict(dict):
    """
    Dictionary subclass that can move tensor values to specified device.

    This utility class extends dict to provide a `to` method that moves
    all tensor values to a specified device while preserving non-tensor values.

    Methods
    -------
    to(device)
        Move all tensor values to the specified device.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        """
        Move all tensor values to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device for tensor values.

        Returns
        -------
        TorchDict
            New dictionary with tensors moved to target device.
        """
        new_dict = TorchDict()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            else:
                new_dict[key] = value
        return new_dict


#  fm2.encode_text(**fm2.processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True))
class OpenClipProcessor(VisionLanguageProcessor):
    """
    Processor for OpenCLIP models.

    Handles preprocessing of images and text for OpenCLIP models,
    including tokenization and image preprocessing.

    Parameters
    ----------
    preprocess : callable
        Image preprocessing function from OpenCLIP.
    tokenizer : callable
        Text tokenizer from OpenCLIP.
    context_length : int, optional
        Maximum context length for text tokenization.
    device : str or torch.device, optional
        Device for tensor operations. Default is "cpu".

    Methods
    -------
    to(device)
        Move processor to specified device.
    __call__(images=None, text=None, **kwargs)
        Process images and/or text inputs.
    """

    def __init__(self, preprocess, tokenizer, context_length=None, device=None):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = "cpu" if device is None else device
        self.context_length = context_length

    def to(self, device):
        """
        Move processor to specified device.

        Parameters
        ----------
        device : str or torch.device
            Target device.
        """
        self.device = device

    def __call__(self, images=None, text=None, **kwargs):
        """
        Process images and/or text inputs for OpenCLIP model.

        Parameters
        ----------
        images : PIL.Image or list of PIL.Image, optional
            Input images to process.
        text : str or list of str, optional
            Input text to tokenize.
        **kwargs
            Additional processing arguments.

        Returns
        -------
        TorchDict
            Dictionary containing processed inputs with keys 'image' and/or 'text'.

        Raises
        ------
        AssertionError
            If neither images nor text is provided.
        """
        outputs = TorchDict()
        assert images is not None or text is not None, "Either images or text must be provided"

        if images:
            if isinstance(images, list):
                images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            else:
                images = self.preprocess(images).to(self.device)
            outputs["image"] = images
            # outputs["pixel_values"] = images  # for compatibility with huggingface
        if text is not None:
            texts = self.tokenizer(
                text, **({"context_length": self.context_length} if self.context_length is not None else {})
            ).to(self.device)
            outputs["text"] = texts

        return outputs


# TODO profile best performance


class HF_Processor(VisionLanguageProcessor):
    """
    Processor for Hugging Face CLIP models.

    Wraps Hugging Face CLIPProcessor for consistent interface.

    Parameters
    ----------
    preprocess : CLIPProcessor
        Hugging Face CLIP processor instance.

    Methods
    -------
    __call__(images=None, text=None, **kwargs)
        Process images and/or text using HF processor.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, images=None, text=None, **kwargs):
        """
        Process inputs using Hugging Face CLIP processor.

        Parameters
        ----------
        images : optional
            Input images.
        text : optional
            Input text.
        **kwargs
            Additional arguments passed to the processor.

        Returns
        -------
        dict
            Processed inputs from HF processor.
        """
        return self.preprocess(text=text, return_tensors="pt", padding=True, images=images)


class HF_Clip(VisionLanguageFoundationModel):
    """
    Hugging Face CLIP model implementation.

    Wraps Hugging Face CLIP models to provide consistent interface
    for vision-language foundation models.

    Parameters
    ----------
    model_url : str
        Hugging Face model identifier or path.

    Attributes
    ----------
    model_url : str
        The model identifier.
    fm : CLIPModel
        The underlying Hugging Face CLIP model.

    Properties
    ----------
    name : str
        Normalized model name for identification.
    processor : HF_Processor
        Processor for input preprocessing.
    device : torch.device
        Current device of the model.

    Methods
    -------
    encode_vision(*args, **kwargs)
        Encode visual inputs into embeddings.
    encode_text(*args, **kwargs)
        Encode text inputs into embeddings.
    forward(*args, **kwargs)
        Forward pass through the model.
    """

    def __init__(self, model_url):
        super().__init__()
        self.model_url = model_url
        self.fm = CLIPModel.from_pretrained(model_url)
        self._processor = HF_Processor(CLIPProcessor.from_pretrained(model_url))  # TODO unify with OpenClipProcessor
        self.fm.eval()

    @property
    def name(self):
        """
        Get normalized model name.

        Returns
        -------
        str
            Model name with special characters replaced by underscores.
        """
        return self.model_url.replace("/", "_").replace("-", "_")

    @property
    def processor(self):
        """
        Get the model processor.

        Returns
        -------
        HF_Processor
            Processor instance for input preprocessing.
        """
        return self._processor

    @property
    def device(self):
        """
        Get current device of the model.

        Returns
        -------
        torch.device
            Current device where model parameters are located.
        """
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_vision(self, *args, **kwargs):
        """
        Encode visual inputs into embeddings.

        Automatically moves inputs to model device before encoding.

        Parameters
        ----------
        *args
            Positional arguments for vision encoding.
        **kwargs
            Keyword arguments for vision encoding.

        Returns
        -------
        torch.Tensor
            Visual embeddings from the model.
        """
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.fm.get_image_features(*args, **kwargs)

    @torch.no_grad()
    def encode_text(self, *args, **kwargs):
        """
        Encode text inputs into embeddings.

        Automatically moves inputs to model device before encoding.

        Parameters
        ----------
        *args
            Positional arguments for text encoding.
        **kwargs
            Keyword arguments for text encoding.

        Returns
        -------
        torch.Tensor
            Text embeddings from the model.
        """
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.fm.get_text_features(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the CLIP model.

        Parameters
        ----------
        *args
            Positional arguments for forward pass.
        **kwargs
            Keyword arguments for forward pass.

        Returns
        -------
        CLIPOutput
            Output from the CLIP model.
        """
        return self.fm(*args, **kwargs)


class OpenClip(VisionLanguageFoundationModel):
    """
    OpenCLIP model implementation.

    Wraps OpenCLIP models to provide consistent interface
    for vision-language foundation models.

    Parameters
    ----------
    model_url : str
        OpenCLIP model identifier.

    Attributes
    ----------
    model_url : str
        The model identifier.
    fm : torch.nn.Module
        The underlying OpenCLIP model.

    Properties
    ----------
    name : str
        Normalized model name for identification.
    processor : OpenClipProcessor
        Processor for input preprocessing.
    device : torch.device
        Current device of the model.

    Methods
    -------
    encode_vision(image)
        Encode visual inputs into embeddings.
    encode_text(text)
        Encode text inputs into embeddings.
    forward(*args, **kwargs)
        Forward pass through the model.
    """

    def __init__(self, model_url):
        super().__init__()
        self.model_url = model_url
        model, preprocess = open_clip.create_model_from_pretrained(model_url)
        tokenizer = open_clip.get_tokenizer(model_url)
        self.fm = model
        self._processor = OpenClipProcessor(preprocess, tokenizer)

    @property
    def name(self):
        """
        Get normalized model name.

        Returns
        -------
        str
            Model name with special characters replaced by underscores.
        """
        return self.model_url.replace("/", "_").replace("-", "_")

    @property
    def processor(self):
        """
        Get the model processor.

        Returns
        -------
        OpenClipProcessor
            Processor instance for input preprocessing.
        """
        return self._processor

    @property
    def device(self):
        """
        Get current device of the model.

        Returns
        -------
        torch.device
            Current device where model parameters are located.
        """
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_vision(self, image: torch.Tensor):
        """
        Encode visual inputs into embeddings.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor. Can be obtained via self.processor(images=image).
            If 3D tensor, will be unsqueezed to add batch dimension.

        Returns
        -------
        torch.Tensor
            Visual embeddings from the model.

        Notes
        -----
        Input can be obtained via self.processor(images=image)
        """
        image = image.unsqueeze(0) if len(image.shape) == 3 else image
        return self.fm.encode_image(image.to(self.device))

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        """
        Encode text inputs into embeddings.

        Parameters
        ----------
        text : torch.Tensor
            Tokenized text tensor. Automatically moved to model device.

        Returns
        -------
        torch.Tensor
            Text embeddings from the model.

        Examples
        --------
        >>> processor_output = model.processor(text="hello")
        >>> embeddings = model.encode_text(processor_output['text'])
        """
        # fm2.encode_text(fm2.processor(text="hallo")['text'])
        text = text.to(self.device)
        return self.fm.encode_text(text)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the OpenCLIP model.

        Parameters
        ----------
        *args
            Positional arguments for forward pass.
        **kwargs
            Keyword arguments for forward pass.

        Returns
        -------
        torch.Tensor or tuple
            Output from the OpenCLIP model.
        """
        return self.fm(*args, **kwargs)
