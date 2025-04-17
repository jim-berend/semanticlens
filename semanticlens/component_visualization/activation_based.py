from collections import namedtuple
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.activation_caching import ActMaxCache

MaxSamples = namedtuple("MaxSamples", ["samples", "activations"])
MaxRefs = namedtuple("MaxRefs", ["sample_ids", "activations"])

DEFAULT_STORAGE = Path("cache") / "concept_examples"


class ActivationComponentVisualizer(AbstractComponentVisualizer):
    def __init__(
        self,
        model,
        dataset,
        layer_names,
        storage_dir=Path("cache") / "concept_examples",
        aggregation_fn="max",
        device=None,
        num_samples=100,
        **kwargs,
    ):
        self.model = model
        self.dataset = dataset
        self._layer_names = layer_names
        self.aggregate_fn = aggregation_fn
        self.num_samples = num_samples
        self.device = device
        self.model.to(self.device)
        self.storage_dir = Path(storage_dir)

        self.actmax_cache = ActMaxCache(self.layer_names, n_collect=self.num_samples, aggregation_fn=self.aggregate_fn)
        self._ran = False

    def run(
        self,
        batch_size=32,
        num_workers=None,
    ):
        try:
            self.actmax_cache = ActMaxCache.load(self.storage_dir)
            print("Cache loaded from ", self.storage_dir)
            self._ran = True
            return
        except FileNotFoundError:
            pass

        device = next(self.model.parameters()).device
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        with self.actmax_cache.hook_context(self.model):
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                _ = self.model(images.to(device)).cpu()

        self.actmax_cache.store(self.storage_dir)
        print("Cache saved at ", self.storage_dir)
        self._ran = True

    def get_max_reference(self, concept_ids: int | list, layer_name: str, n_ref: int, batch_size: int = 32):
        raise NotImplementedError(
            "`get_max_reference` is not yet implemented for {self.__class__.__name__} but will be available soon."
        )
        # [ ] TODO act/grad-based cropping lxt like approach?
        # r_range = (0, n_ref) if isinstance(n_ref, int) else n_ref
        results = {}
        for i, (ids, acts) in tqdm(self.get_max(concept_ids, layer_name).items()):
            # samples = [to_pil_image(self.dataset[i][0])  if return_pil else self.dataset[i][0] for i in ids]
            samples = [to_pil_image(self.dataset[i][0]) for i in ids]
            results[i] = MaxSamples(samples, acts)

        return results

    def get_act_max_sample_ids(self, layer_name: str):
        return self.actmax_cache.cache[layer_name].sample_ids

    def __repr__(self):
        return (
            "ActBasedFeatureVisualization("
            + f"\n\tmodel={self.model.__class__.__name__},"
            + f"\n\tdataset={self.dataset.__class__.__name__},"
            + f"\n\tstorage_dir={self.storage_dir},"
            + f"\n\taggregation_fn={self.aggregate_fn},"
            + f"\n\tactmax_cache={self.actmax_cache},\n)"
        )

    def to(self, device: torch.device | str):
        """Move model to device"""
        self.model.to(device)
        self.device = device
        return self

    @property
    def metadata(self) -> dict:
        """Returns the metadata of the visualizer."""
        return {
            "aggregation_fn": self.aggregate_fn,
            "actmax_cache": repr(self.actmax_cache),
        }
