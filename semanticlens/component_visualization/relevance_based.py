import os

import torch
from crp.concepts import ChannelConcept as Concept
from crp.helper import load_maximization
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat as Composite

from semanticlens.component_visualization.base import AbstractComponentVisualizer
from semanticlens.utils.render import crop_and_mask_images


class RelevanceComponentVisualizer(FeatureVisualization, AbstractComponentVisualizer):
    def __init__(
        self,
        attribution,
        dataset,
        layer_names,
        preprocess_fn,
        composite=None,
        aggregation_fn="sum",
        abs_norm=True,
        storage_dir="FeatureVisualization",
        device=None,
        num_samples=100,
        cache=None,
        plot_fn=crop_and_mask_images,
    ):
        layer_names = [layer_names] if not isinstance(layer_names, list) else layer_names
        super().__init__(
            attribution,
            dataset,
            layer_map={layer_name: Concept() for layer_name in layer_names},
            preprocess_fn=preprocess_fn,
            max_target=aggregation_fn,
            abs_norm=abs_norm,
            path=storage_dir,
            device=device,
            cache=cache,
        )
        self._layer_names = layer_names

        self.num_samples = num_samples
        self.composite = composite
        self.storage_dir = storage_dir
        self.plot_fn = plot_fn
        self.aggregation_fn = aggregation_fn
        self.abs_norm = abs_norm

        from crp.maximization import Maximization
        from crp.statistics import Statistics

        # set normalization to false
        self.ActMax = Maximization(mode="activation", max_target=aggregation_fn, abs_norm=False, path=self.storage_dir)
        self.ActStats = Statistics(mode="activation", max_target=aggregation_fn, abs_norm=False, path=self.storage_dir)

        self.ActMax.SAMPLE_SIZE = self.num_samples

        self._ran = self.check_if_preprocessed()

    def run(
        self,
        composite: Composite = None,
        data_start=0,
        data_end=None,
        batch_size=32,
        checkpoint=500,
        on_device=None,
        **kwargs,
    ):
        composite = self._check_composite(composite)
        if not self.check_if_preprocessed():
            print("Preprocessing...")
            data_end = len(self.dataset) if data_end is None else data_end
            results = super().run(composite, data_start, data_end, batch_size, checkpoint, on_device)
            self._ran = True
            return results

        else:
            print("Already preprocessed")
            return [j for j in os.listdir(self.ActMax.PATH) if any([l in j for l in self.layer_names])]

    @torch.enable_grad()  # required for LRP/CRP
    def get_max_reference(self, concept_ids: int | list, layer_name: str, n_ref: int, batch_size: int = 32):
        mode = "activation"
        r_range = (0, n_ref)
        composite = self.composite
        plot_fn = self.plot_fn
        rf = True
        try:
            return super().get_max_reference(concept_ids, layer_name, mode, r_range, composite, rf, plot_fn, batch_size)
        except AttributeError as e:
            print("Error during LRP/CRP-based concept-visualization.")
            print("Note `crp` requires gradients: Make sure to execute with torch autograd enabled.")
            raise e

    def check_if_preprocessed(self):
        return bool(os.listdir(self.ActMax.PATH)) and all(
            any([i.startswith(layer_name) for i in os.listdir(self.ActMax.PATH)]) for layer_name in self.layer_names
        )

    def _check_composite(self, composite):
        assert composite or self.composite, "Composite must be provided or set in initialization (__init__)"
        return composite or self.composite

    def get_act_max_sample_ids(self, layer_name: str):
        return torch.tensor(load_maximization(path_folder=self.ActMax.PATH, layer_name=layer_name)[0]).T

    def to(self, device: torch.device | str):
        self.device = device
        self.attribution.model.to(self.device)

    @property
    def metadata(self) -> dict:
        return {
            "preprocess_fn": str(self.preprocess_fn),
            "abs_norm": self.abs_norm,
            "aggregation_fn": self.aggregation_fn,
            "storage_dir": str(self.storage_dir),
            "device": str(self.device),
            "num_samples": self.num_samples,
            "plot_fn": str(self.plot_fn),
        }
