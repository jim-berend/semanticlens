# tests/foundation_models/test_clip.py

import pytest
import torch
from PIL import Image

# Import the classes to be tested from your package
from semanticlens.foundation_models import ClipMobile, OpenClip, SigLipV2

# A list of all specific model classes to test
MODEL_CLASSES = [SigLipV2, ClipMobile]


# A fixture to provide a dummy image for the tests
# This avoids creating the image repeatedly in each test function
@pytest.fixture(scope="module")
def dummy_image():
    """Creates a simple 64x64 RGB PIL Image."""
    return Image.new("RGB", (64, 64), color="red")


# A fixture to provide sample text
@pytest.fixture(scope="module")
def sample_text():
    """Returns a sample list of text prompts."""
    return ["a red square", "a photo of a cat"]


# --- Tests ---


def test_open_clip_initialization_and_encoding(dummy_image, sample_text):
    """
    Tests the base OpenClip class with a specific small model.
    This ensures the fundamental encoding logic works as expected.
    """
    # Use a small, well-known model for quick testing
    model_url = "ViT-B-32-quickgelu"

    fm = OpenClip(url=model_url, device="cpu", load_weights=False)

    # 1. Test image encoding
    img_input = fm.preprocess(dummy_image)
    assert img_input.ndim == 4, "Preprocessed image should have a batch dimension"

    img_features = fm.encode_image(img_input)
    assert isinstance(img_features, torch.Tensor)
    assert img_features.shape[0] == 1
    assert img_features.ndim == 2

    # 2. Test text encoding
    text_input = fm.tokenize(sample_text)
    text_features = fm.encode_text(text_input)
    assert isinstance(text_features, torch.Tensor)
    assert text_features.shape[0] == len(sample_text)
    assert text_features.ndim == 2

    # 3. Check feature dimension consistency
    assert img_features.shape[1] == text_features.shape[1]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_specialized_clip_models(model_class, dummy_image, sample_text):
    """
    Tests specialized CLIP variants (SigLipV2, ClipMobile).
    This test verifies that they can be initialized and perform encoding.
    """
    fm = model_class(device="cpu", load_weights=False)

    # 1. Test image encoding
    img_input = fm.preprocess(dummy_image)
    img_features = fm.encode_image(img_input)
    assert isinstance(img_features, torch.Tensor)
    assert img_features.shape[0] == 1
    assert img_features.ndim == 2

    # 2. Test text encoding
    text_input = fm.tokenize(sample_text)
    text_features = fm.encode_text(text_input)
    assert isinstance(text_features, torch.Tensor)
    assert text_features.shape[0] == len(sample_text)
    assert text_features.ndim == 2

    # 3. Check feature dimension consistency
    assert img_features.shape[1] == text_features.shape[1]


def test_device_management():
    """
    Tests the `.to()` method and `.device` property of a foundation model.
    Skips the test if no CUDA device is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device management test")

    fm = OpenClip(url="ViT-B-32-quickgelu", device="cpu", load_weights=False)
    assert "cpu" in str(fm.device)

    fm.model.to("cuda")

    assert "cuda" in str(fm.device)
