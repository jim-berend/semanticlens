"""
ConceptTensor: A tensor subclass for storing embeddings with associated metadata.

This module provides a ConceptTensor class that extends PyTorch tensors to include
metadata storage capabilities, enabling rich semantic information to be preserved
alongside tensor data.
"""

import json
import logging

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


class ConceptTensor(torch.Tensor):
    """
    A tensor subclass that stores embeddings with associated metadata.

    ConceptTensor extends PyTorch tensors to include metadata storage and
    serialization capabilities using safetensors format. This enables semantic
    information about concepts to be preserved alongside the tensor data.

    Parameters
    ----------
    data : array-like
        Tensor data to be stored.
    metadata : dict, optional
        Dictionary containing metadata about the concept tensor.
        Default is an empty dictionary.

    Attributes
    ----------
    metadata : dict
        Dictionary containing metadata about the tensor.

    Examples
    --------
    >>> import torch
    >>> data = torch.randn(10, 512)
    >>> metadata = {"concept": "animals", "model": "clip"}
    >>> concept_tensor = ConceptTensor(data, metadata=metadata)
    >>> concept_tensor.shape
    torch.Size([10, 512])
    >>> concept_tensor.metadata
    {'concept': 'animals', 'model': 'clip'}
    """

    def __new__(cls, data, metadata=None):
        obj = torch.as_tensor(data).as_subclass(cls)
        obj._metadata = metadata or {}
        return obj

    def __init__(self, data, metadata=None):
        pass  # Already handled in __new__

    @property
    def metadata(self):
        """
        Get metadata associated with the tensor.

        Returns
        -------
        dict or None
            Dictionary containing metadata, or None if no metadata exists.
        """
        if not hasattr(self, "_metadata"):
            return None
        return self._metadata

    def save(self, filepath):
        """
        Save tensor and metadata to disk using safetensors format.

        Parameters
        ----------
        filepath : str or Path
            Path where the tensor should be saved.

        Examples
        --------
        >>> concept_tensor.save("my_concepts.safetensors")
        ConceptTensor saved at: my_concepts.safetensors
        """
        # Convert metadata values to strings (safetensors only supports str)
        encoded_metadata = {k: json.dumps(v) for k, v in self.metadata.items()}
        save_file({"tensor": self.clone()}, filepath, metadata=encoded_metadata)
        logger.info("ConceptTensor saved at: %s", filepath)

    @classmethod
    def load(cls, filepath, device="cpu"):
        """
        Load tensor and metadata from disk using safetensors format.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved tensor file.
        device : str, default="cpu"
            Device on which to load the tensor.

        Returns
        -------
        ConceptTensor
            Loaded tensor with metadata.

        Examples
        --------
        >>> loaded_tensor = ConceptTensor.load("my_concepts.safetensors")
        >>> loaded_tensor.metadata
        {'concept': 'animals', 'model': 'clip'}
        """
        with safe_open(filepath, framework="pt", device=device) as f:
            tensor = f.get_tensor("tensor")
            # Parse metadata values back from JSON strings
            metadata = {k: json.loads(v) for k, v in f.metadata().items()}
        return cls(tensor, metadata=metadata)

    def __repr__(self):
        """
        Return string representation including metadata.

        Returns
        -------
        str
            String representation of the tensor and its metadata.
        """
        if hasattr(self, "metadata"):
            metadata_str = json.dumps(self.metadata, indent=0, sort_keys=True)
        else:
            metadata_str = "<None>"
        metadata_str = metadata_str.replace("\n", " ")
        if len(metadata_str) > 60:
            metadata_str = metadata_str[:60] + "...}"
        return f"{super().__repr__()[:-1]},\n" + " " * 14 + f"metadata={metadata_str})"


"""Concept tensors module for tensor container and metadata management.

This module provides utilities for working with tensors and their associated metadata,
including serialization, deserialization, and efficient caching capabilities using
safetensors format.

Classes
-------
ConceptTensors
    A robust container for tensors and their associated metadata.

Functions
---------
metadata
    Extracts metadata from safetensor bytes.
"""

import struct
from logging import getLogger
from pathlib import Path

import torch
from safetensors.torch import load, load_file, save

logger = getLogger(__name__)


def metadata(safetensor_bytes: bytes) -> dict:
    """Extract metadata from safetensor bytes.

    Parameters
    ----------
    safetensor_bytes : bytes
        The raw safetensor bytes containing header and metadata information.

    Returns
    -------
    dict
        The metadata dictionary extracted from the safetensor header.
        Returns an empty dict if no metadata is found.

    Notes
    -----
    This function parses the safetensor format header to extract metadata.
    The header contains a size field followed by JSON-encoded metadata.
    """
    header_size = struct.unpack("<Q", safetensor_bytes[:8])[0]
    header_bytes = safetensor_bytes[8 : 8 + header_size]
    header = json.loads(header_bytes)
    return header.get("metadata", {})


class ConceptTensors:
    """A robust container for tensors and their associated metadata.

    This class is the standard data structure for representing concepts.
    It supports explicit saving/loading via .save() and .load() methods,
    and is also designed to be efficiently cached by joblib.

    Parameters
    ----------
    tensors : dict[str, torch.Tensor] or torch.Tensor
        Dictionary mapping tensor names to PyTorch tensors, or a single tensor
        (which will be stored as "main").
    metadata : dict
        Dictionary containing metadata associated with the tensors.

    Attributes
    ----------
    tensors : dict[str, torch.Tensor]
        Dictionary containing all tensors in this concept.
    metadata : dict
        Metadata dictionary associated with the tensors.

    Examples
    --------
    >>> import torch
    >>> tensor = torch.randn(10, 5)
    >>> metadata = {"source": "example", "version": 1}
    >>> concept = ConceptTensors(tensor, metadata)
    >>> concept.shape
    torch.Size([10, 5])

    >>> # With multiple tensors
    >>> tensors = {"embeddings": torch.randn(10, 5), "weights": torch.randn(5,)}
    >>> concept = ConceptTensors(tensors, metadata)
    """

    def __init__(self, tensors: dict[str, torch.Tensor], metadata: dict):
        """Initialize a ConceptTensors instance.

        Parameters
        ----------
        tensors : dict[str, torch.Tensor] or torch.Tensor
            Dictionary mapping tensor names to PyTorch tensors, or a single tensor
            (which will be stored as "main").
        metadata : dict
            Dictionary containing metadata associated with the tensors.
        """
        if not isinstance(tensors, dict):
            # To ease transition, we can accept a single tensor
            tensors = {"main": tensors}
        self.tensors = tensors
        self.metadata = metadata

    @property
    def main(self) -> torch.Tensor:
        """Get the primary tensor.

        Returns
        -------
        torch.Tensor
            The tensor stored under the "main" key.

        Raises
        ------
        KeyError
            If no tensor is stored under the "main" key.
        """
        return self.tensors["main"]

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the primary tensor.

        Returns
        -------
        torch.Size
            The shape of the primary ("main") tensor.

        Raises
        ------
        KeyError
            If no tensor is stored under the "main" key.
        """
        return self.main.shape

    @property
    def device(self) -> torch.device:
        """Get the device of the primary tensor.

        Returns
        -------
        torch.device
            The device on which the primary ("main") tensor is located.

        Raises
        ------
        KeyError
            If no tensor is stored under the "main" key.
        """
        return self.main.device

    def to(self, device: str | torch.device):
        """Move all contained tensors to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The target device to move tensors to (e.g., 'cpu', 'cuda', 'cuda:0').

        Returns
        -------
        ConceptTensors
            A new ConceptTensors instance with all tensors moved to the specified device.
            The metadata is preserved unchanged.

        Examples
        --------
        >>> concept_cpu = concept.to('cpu')
        >>> concept_gpu = concept.to('cuda:0')
        """
        new_tensors = {k: v.to(device) for k, v in self.tensors.items()}
        return ConceptTensors(new_tensors, self.metadata)

    def save(self, filepath: str | Path):
        """Save the ConceptTensors to a file using safetensors format.

        Parameters
        ----------
        filepath : str or Path
            The path where the ConceptTensors will be saved. The file will be
            saved in safetensors format with embedded metadata.

        Notes
        -----
        The metadata is JSON-encoded and embedded in the safetensors file header.
        This allows for efficient loading of both tensors and metadata.

        Examples
        --------
        >>> concept.save('my_concept.safetensors')
        >>> concept.save(Path('data/concepts/concept_001.safetensors'))
        """
        encoded_meta = {k: json.dumps(v) for k, v in self.metadata.items()}
        save_file(self.tensors, filepath, metadata=encoded_meta)
        print(f"Concept saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path, device="cpu"):
        """Load a ConceptTensors from a safetensors file.

        Parameters
        ----------
        filepath : str or Path
            The path to the safetensors file to load.
        device : str or torch.device, optional
            The device to load tensors onto (default: "cpu").

        Returns
        -------
        ConceptTensors
            A new ConceptTensors instance loaded from the file.

        Examples
        --------
        >>> concept = ConceptTensors.load('my_concept.safetensors')
        >>> concept_gpu = ConceptTensors.load('my_concept.safetensors', device='cuda')

        Notes
        -----
        This method loads both the tensors and metadata from the safetensors file.
        The metadata is automatically JSON-decoded during the loading process.
        """
        tensors = {}
        with load_file(filepath, device=device) as f:
            # Safely load metadata first
            loaded_meta = {k: json.loads(v) for k, v in f.metadata().items()}
            # Load all tensors
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return cls(tensors, loaded_meta)

    # --- Methods for Joblib Caching ---
    def __getstate__(self) -> dict:
        """Get serializable state for joblib/pickle caching.

        Returns
        -------
        dict
            A dictionary containing the serialized safetensor bytes.
            The tensors and metadata are encoded into a single bytes object
            for efficient serialization.

        Notes
        -----
        This method is called automatically by joblib and pickle during
        serialization. The tensors and metadata are converted to safetensor
        format in memory for efficient storage and retrieval.
        """
        logger.info("got called!")
        encoded_meta = {k: json.dumps(v) for k, v in self.metadata.items()}
        # Serialize to in-memory bytes
        safetensor_bytes = save(self.tensors, metadata=encoded_meta)
        return {"safetensor_bytes": safetensor_bytes}

    def __setstate__(self, state: dict):
        """Restore state from joblib/pickle deserialization.

        Parameters
        ----------
        state : dict
            The state dictionary returned by __getstate__, containing
            serialized safetensor bytes.

        Notes
        -----
        This method is called automatically by joblib and pickle during
        deserialization. It reconstructs the tensors and metadata from
        the safetensor bytes format.
        """
        logger.info("got called!")
        safetensor_bytes = state["safetensor_bytes"]
        # Load from in-memory bytes
        loaded_tensors = load(safetensor_bytes)
        loaded_meta = metadata(safetensor_bytes)
        self.tensors = loaded_tensors
        self.metadata = {k: json.loads(v) for k, v in loaded_meta.items()}

    def __repr__(self):
        """Return a string representation of the ConceptTensors.

        Returns
        -------
        str
            A human-readable string representation showing tensor keys,
            shape of the main tensor, and a truncated view of metadata.

        Examples
        --------
        >>> concept = ConceptTensors(torch.randn(10, 5), {"source": "test"})
        >>> repr(concept)
        'Concept(tensors=[\'main\'], shape=torch.Size([10, 5]), metadata={\'source\': \'test\'})'
        """
        from pprint import pformat

        meta_str = str(self.metadata)
        if len(meta_str) > 100:
            meta_str = meta_str[:100] + "...}"
        return f"ConceptTensors(tensors={pformat(self.tensors)}, shape={self.shape}, metadata={meta_str})"


if __name__ == "__main__":
    vec2 = ConceptTensor([1.0, 2.0, 3.0], metadata={"concept": "text", "tags": ["book", "shelf"]})
    vec = ConceptTensor([1.0, 2.0, 3.0], metadata={"concept": "emotion", "tags": ["joy", "calm"]})
    vec.save("vec.safetensors")

    loaded_vec = ConceptTensor.load("vec.safetensors")
    print(loaded_vec)
    print("Metadata:", loaded_vec.metadata)
    print("Type:", type(loaded_vec))
    print(vec, vec.device)
    vec = vec.to("cuda")
    print(vec, vec.device)
    print(ConceptTensor(torch.rand(12, 1221)).cuda())
