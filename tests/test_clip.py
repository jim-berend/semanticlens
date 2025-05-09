import pytest
import torch

from semanticlens.foundation_models.clip import HF_Clip, OpenClip


def load_dummy_image_via_request():
    # Load a dummy image from the internet
    from io import BytesIO

    import requests
    from PIL import Image

    url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    return img


@pytest.fixture
def dummy_image():
    return load_dummy_image_via_request()


@pytest.fixture
def dummy_text():
    return "a photo of a cat"


@pytest.mark.parametrize(
    "model_class, model_url",
    [
        (HF_Clip, "openai/clip-vit-base-patch16"),
        (OpenClip, "hf-hub:apple/MobileCLIP-S2-OpenCLIP"),
    ],
)
def test_model_initialization(model_class, model_url):
    model = model_class(model_url)
    assert model is not None
    assert model.processor is not None


@pytest.mark.parametrize(
    "model_class, model_url",
    [
        (HF_Clip, "openai/clip-vit-base-patch16"),
        (OpenClip, "hf-hub:apple/MobileCLIP-S2-OpenCLIP"),
    ],
)
def test_text_encoding(model_class, model_url, dummy_text):
    model = model_class(model_url)
    encoded_text = model.encode_text(**model.processor(text=dummy_text))
    assert isinstance(encoded_text, torch.Tensor)
    assert encoded_text.shape[0] > 0  # Ensure some output is generated


@pytest.mark.parametrize(
    "model_class, model_url",
    [
        (HF_Clip, "openai/clip-vit-base-patch16"),
        (OpenClip, "hf-hub:apple/MobileCLIP-S2-OpenCLIP"),
    ],
)
def test_image_encoding(model_class, model_url, dummy_image):
    model = model_class(model_url)
    encoded_image = model.encode_vision(**model.processor(images=dummy_image))
    assert isinstance(encoded_image, torch.Tensor)
    assert encoded_image.shape[0] > 0  # Ensure some output is generated


@pytest.mark.parametrize(
    "model_class, model_url",
    [
        (HF_Clip, "openai/clip-vit-base-patch16"),
        (OpenClip, "hf-hub:apple/MobileCLIP-S2-OpenCLIP"),
    ],
)
def test_model_device_transfer(model_class, model_url, dummy_text, dummy_image):
    model = model_class(model_url)
    model.to("cuda")
    assert next(model.parameters()).device.type == "cuda"

    encoded_text = model.encode_text(**model.processor(text=dummy_text))
    encoded_image = model.encode_vision(**model.processor(images=dummy_image))

    assert encoded_text.device.type == "cuda"
    assert encoded_image.device.type == "cuda"
