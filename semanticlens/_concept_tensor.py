import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class ConceptTensor(torch.Tensor):
    def __new__(cls, data, metadata=None):
        obj = torch.as_tensor(data).as_subclass(cls)
        obj._metadata = metadata or {}
        return obj

    def __init__(self, data, metadata=None):
        pass  # Already handled in __new__

    @property
    def metadata(self):
        """Metadata property"""
        if not hasattr(self, "_metadata"):
            return None
        return self._metadata

    def save(self, filepath):
        """Store tensor and metadata using safetensors"""
        # Convert metadata values to strings (safetensors only supports str)
        encoded_metadata = {k: json.dumps(v) for k, v in self.metadata.items()}
        save_file({"tensor": self.clone()}, filepath, metadata=encoded_metadata)
        print("ConceptTensor saved at: ", filepath)  # [ ] TODO replace with proper logging?

    @classmethod
    def load(cls, filepath, device="cpu"):
        """Load tensor and metadata using safetensors"""
        with safe_open(filepath, framework="pt", device=device) as f:
            tensor = f.get_tensor("tensor")
            # Parse metadata values back from JSON strings
            metadata = {k: json.loads(v) for k, v in f.metadata().items()}
        return cls(tensor, metadata=metadata)

    def __repr__(self):
        """Custom representation to include metadata"""
        if hasattr(self, "metadata"):
            metadata_str = json.dumps(self.metadata, indent=0, sort_keys=True)
        else:
            metadata_str = "<None>"
        metadata_str = metadata_str.replace("\n", " ")
        if len(metadata_str) > 60:
            metadata_str = metadata_str[:60] + "...}"
        return f"{super().__repr__()[:-1]},\n" + " " * 14 + f"metadata={metadata_str})"


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
