import os
import tempfile

import torch

import semanticlens as sl


def test_concept_tensor_save_and_load():
    data = [1.0, 2.0, 3.0]
    metadata = {"concept": "emotion", "tags": ["joy", "calm"]}
    tensor = sl.ConceptTensor(data, metadata=metadata)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as temp_file:
        filepath = temp_file.name
        tensor.save(filepath)

    try:
        assert os.path.exists(filepath), "File should be saved at the specified path"

        loaded_tensor = sl.ConceptTensor.load(filepath)

        assert isinstance(loaded_tensor, sl.ConceptTensor), "Loaded object should be an instance of ConceptTensor"
        assert torch.equal(loaded_tensor, torch.tensor(data)), "Loaded tensor data should match the original data"
        assert loaded_tensor.metadata == metadata, "Loaded metadata should match the original metadata"
    finally:
        os.remove(filepath)  # Ensure cleanup happens even if an assertion fails
