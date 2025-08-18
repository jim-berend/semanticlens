# tests/test_lens.py

import pytest
import torch

from semanticlens.lens import Lens


# Use mocker to create fake versions of the abstract classes
@pytest.fixture
def mock_fm(mocker):
    """Mocks the AbstractVLM (foundation model)."""
    fm = mocker.MagicMock()
    fm.device = "cpu"
    # Simulate text embedding
    fm.encode_text.return_value = torch.randn(1, 128)
    # Return self for chained calls like fm.to(device)
    fm.to.return_value = fm
    return fm


@pytest.fixture
def mock_cv(mocker, tmp_path):
    """Mocks the AbstractComponentVisualizer."""
    # Mock the storage directory and metadata needed for caching logic
    cv = mocker.MagicMock()
    cv.caching = True
    cv.storage_dir = tmp_path
    # Simulate the private method that Lens calls
    cv._compute_concept_db.return_value = {"layer1": torch.randn(10, 5, 128)}
    cv.metadata = {"model": "test_model"}
    return cv


def test_lens_initialization(mock_fm):
    """
    Tests that the Lens class initializes correctly and moves the
    foundation model to the specified device.
    """
    lens = Lens(fm=mock_fm, device="cpu")
    assert lens.fm is mock_fm
    mock_fm.to.assert_called_with("cpu")


def test_compute_concept_db_no_cache(mock_fm, mock_cv, mocker):
    """
    Tests that _compute_concept_db is called when the cache file does not exist.
    """
    # Mock Path.exists to return False
    # mocker.patch("pathlib.Path.exists", return_value=False)
    # We don't need to mock save_file, just ensure the computation is called
    mock_save = mocker.patch("semanticlens.lens.save_file")

    lens = Lens(fm=mock_fm)
    concept_db = lens.compute_concept_db(mock_cv)

    # Assert that the computation was performed
    mock_cv._compute_concept_db.assert_called_once_with(mock_fm)
    # Assert that the result was saved to cache
    mock_save.assert_called_once()
    assert "layer1" in concept_db


def test_compute_concept_db_with_cache(mock_fm, mock_cv, mocker):
    """
    Tests that the concept_db is loaded from cache if the file exists.
    """
    concept_db_dir = mock_cv.storage_dir / "concept_database"
    concept_db_dir.mkdir()
    # Mock Path.exists to return True
    mocker.patch("pathlib.Path.exists", return_value=True)
    mock_load = mocker.patch("semanticlens.lens.load_file", return_value={"layer1": "data_from_cache"})

    lens = Lens(fm=mock_fm)
    concept_db = lens.compute_concept_db(mock_cv)

    # Assert that the data was loaded from the cache file
    mock_load.assert_called_once()
    # Assert that computation was *not* performed
    mock_cv._compute_concept_db.assert_not_called()
    assert concept_db["layer1"] == "data_from_cache"


def test_lens_text_probing(mock_fm):
    """
    Tests the text_probing method to ensure it calls the fm correctly.
    """
    lens = Lens(fm=mock_fm)
    aggregated_db = {"layer1": torch.randn(10, 128)}

    # The actual probing is a stateless function, so we just check the call
    results = lens.text_probing("a test query", aggregated_db)

    mock_fm.encode_text.assert_called_once()
    assert "layer1" in results
    assert results["layer1"].shape[0] == 1  # 1 query
    assert results["layer1"].shape[1] == 10  # 10 neurons
