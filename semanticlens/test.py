# %%
from __future__ import annotations

import logging

import torch
from torch import nn

from semanticlens import Lens
from semanticlens import foundation_models as FMs
from semanticlens.component_visualization import aggregators
from semanticlens.component_visualization.activation_based import ActivationComponentVisualizer as ActConceptVisualizer

logger = logging.getLogger(__name__)

ConceptTensors = dict
AbstractFM = nn.Module


def main():
    # loading model and dataset and testing everything

    device = "cuda"

    # load model
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model_name = "hf_hub:timm/test_resnet.r160_in1k"
    # model_name = "hf_hub:timm/vit_small_patch14_dinov2.lvd142m"  # "hf_hub:timm/vit_large_patch14_dinov2.lvd142m"
    # model_name = "hf_hub:timm/vit_small_patch14_dinov2.lvd142m"  # "hf_hub:timm/vit_large_patch14_dinov2.lvd142m"
    # layer_names = ["blocks.11"]
    # model_name = "resnet50d.a1_in1k"
    layer_names = ["layer4"]
    # model_name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    # layer_names = ["blocks.11"]
    # model_name = "hf_hub:timm/tiny_vit_21m_224.dist_in22k_ft_in1k"
    # layer_names = ["stages.3"]
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    config = resolve_data_config({}, model=model)
    model.transform = create_transform(**config)
    model.name = model_name

    # loading dataset
    from datafiles import get_dataset

    dataset_name = "imagenet"
    dataset_model = get_dataset(dataset_name)(
        data_path="/data/datasets/ImageNet-complete",
        split="val",
        transform=model.transform,  # TODO use different way of dealing with transform!
    )
    dataset_model.name = dataset_name

    def get_unnormalization_transform(
        mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]),
        std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]),
    ) -> torch.nn.Module:
        """Return a transform to undo the normalization of an image tensor.

        This function creates a composition of transforms that reverses standard
        ImageNet normalization, which is useful for visualization purposes.

        Parameters
        ----------
        mean : torch.Tensor, optional
            Mean values used in the original normalization. Default is ImageNet means
            [0.485, 0.456, 0.406] for RGB channels.
        std : torch.Tensor, optional
            Standard deviation values used in the original normalization. Default is
            ImageNet stds [0.229, 0.224, 0.225] for RGB channels.

        Returns
        -------
        torch.nn.Module
            A composed transform that reverses the normalization when applied to a tensor.
        """
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
                transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
            ]
        )

    from torchvision import transforms

    try:
        _nrm = [t for t in model.transform.transforms if isinstance(t, transforms.Normalize)][0]
        kwargs = {"mean": _nrm.mean, "std": _nrm.std}
    except IndexError:
        kwargs = dict()

    dataset_model.invert_normalization = get_unnormalization_transform(
        **kwargs
        # mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])
    )

    from torchvision.transforms import Compose, Normalize

    print(model.transform)

    no_normaliztion_transform = Compose([t for t in model.transform.transforms if not isinstance(t, Normalize)])
    print(no_normaliztion_transform)

    dataset_fm = get_dataset(dataset_name)(
        data_path="/data/datasets/ImageNet-complete",
        split="val",
        transform=None,  # no_normaliztion_transform,  # TODO use different way of dealing with transform!
    )
    dataset_fm.name = dataset_name

    # putting everything together

    cache_dir = "./cache"

    cv = ActConceptVisualizer(
        model,
        dataset_model,
        dataset_fm,
        num_samples=15,
        layer_names=(layer_names),
        device=device,
        aggregate_fn=aggregators.aggregate_conv_mean,
        cache_dir=cache_dir,
    )

    # fm = FMs.ClipMobile(device=device)
    # fm = FMs.SigLipV2(device=device)
    fm = FMs.OpenClip(url="hf-hub:timm/ViT-B-16-SigLIP", device=device)

    lens = Lens(fm, device=device)

    concept_db = lens.compute_concept_db(cv, batch_size=128, num_workers=32)

    cv.visualize_components(component_ids=torch.arange(10), layer_name=layer_names[-1])

    out = lens.text_probing(
        "dog dog dog", {k: v.to(device).mean(1) for k, v in concept_db.items()}, templates=["a i tap of {}"]
    )  # TODO handle device properly
    cv.visualize_components(component_ids=out[layer_names[-1]].topk(10).indices, layer_name=layer_names[-1])

    def load_dummy_image_via_request():
        # Load a dummy image from the internet
        from io import BytesIO

        import requests
        from PIL import Image

        url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTUngqKT_7djRFbbaSqLrDkAj6e68K6_E8XPtrliqmhXdkUhTjpLWNpI_m7QqdVuiWPRm-JNwKHLfb5P93RpDO6GA"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        return img

    img = load_dummy_image_via_request()

    im_out = lens.image_probing([img] * 50, {k: v.to("cuda").mean(1) for k, v in concept_db.items()})

    cv.visualize_components(component_ids=im_out[layer_names[-1]].topk(10).indices, layer_name=layer_names[-1])

    print("debug")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)
    # from semanticlens.log_setup import setup_colored_logging

    # setup_colored_logging()
    main()
