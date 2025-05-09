import open_clip
import torch
from transformers import CLIPModel, CLIPProcessor

from semanticlens.foundation_models.base import VisionLanguageFoundationModel, VisionLanguageProcessor


class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super(TorchDict, self).__init__(*args, **kwargs)

    def to(self, device):
        new_dict = TorchDict()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            else:
                new_dict[key] = value
        return new_dict


#  fm2.encode_text(**fm2.processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True))
class OpenClipProcessor(VisionLanguageProcessor):
    def __init__(self, preprocess, tokenizer, context_length=None, device=None):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = "cpu" if device is None else device
        self.context_length = context_length

    def to(self, device):
        self.device = device

    def __call__(self, images=None, text=None, **kwargs):
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
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, images=None, text=None, **kwargs):
        return self.preprocess(text=text, return_tensors="pt", padding=True, images=images)


class HF_Clip(VisionLanguageFoundationModel):
    def __init__(self, model_url):
        super().__init__()
        self.model_url = model_url
        self.fm = CLIPModel.from_pretrained(model_url)
        self._processor = HF_Processor(CLIPProcessor.from_pretrained(model_url))  # TODO unify with OpenClipProcessor
        self.fm.eval()

    @property
    def name(self):
        return self.model_url.replace("/", "_").replace("-", "_")

    @property
    def processor(self):
        return self._processor

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_vision(self, *args, **kwargs):
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.fm.get_image_features(*args, **kwargs)

    @torch.no_grad()
    def encode_text(self, *args, **kwargs):
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.fm.get_text_features(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.fm(*args, **kwargs)


class OpenClip(VisionLanguageFoundationModel):
    def __init__(self, model_url):
        super().__init__()
        self.model_url = model_url
        model, preprocess = open_clip.create_model_from_pretrained(model_url)
        tokenizer = open_clip.get_tokenizer(model_url)
        self.fm = model
        self._processor = OpenClipProcessor(preprocess, tokenizer)

    @property
    def name(self):
        return self.model_url.replace("/", "_").replace("-", "_")

    @property
    def processor(self):
        return self._processor

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_vision(self, image: torch.Tensor):
        """input can be obtained via self.processor(images=image)"""
        image = image.unsqueeze(0) if len(image.shape) == 3 else image
        return self.fm.encode_image(image.to(self.device))

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        # fm2.encode_text(fm2.processor(text="hallo")['text'])
        text = text.to(self.device)
        return self.fm.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.fm(*args, **kwargs)
