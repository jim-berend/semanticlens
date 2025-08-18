Quickstart
==========

This quickstart shows the bare‑minimum steps to run **SemanticLens** on a vision model. It mirrors the workflow from the Quickstart notebook but keeps details to a minimum. For the full, step‑by‑step tutorial (with explanations, options, and visuals), use the notebook.


Install
-------
Ensure the required packages are available. At minimum you'll need `semanticlens` and its common dependencies.

.. code-block:: bash

   pip install semanticlens timm torchvision
   # PyTorch must be installed separately according to your platform/CUDA setup.



Import
------

.. code-block:: python

   import torch
   import timm
   import semanticlens as sl
   from semanticlens.component_visualization import aggregators, ActivationComponentVisualizer
   from semanticlens.foundation_models import ClipMobile
   from torchvision.datasets import ImageFolder
   from torchvision.transforms import v2 as transforms


Minimal example
---------------

.. code-block:: python

   # Select device
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # 1) Load the model to analyze (from timm) and its transforms
   model_name = "resnet50d.a1_in1k"
   model = timm.create_model(model_name, pretrained=True).to(device).eval()
   model.name = model_name  # used for caching
   data_config = timm.data.resolve_data_config({}, model=model)
   model_transform = timm.data.create_transform(**data_config)

   # 2) Define a transform for the foundation model (no normalization here)
   fm_transform = transforms.Compose([
       transforms.Resize(data_config["input_size"][1], interpolation=transforms.InterpolationMode.BICUBIC),
       transforms.CenterCrop(data_config["input_size"][1]),
   ])

   # 3) Point both datasets at the SAME image collection
   #    (use your own dataset root; ImageNet used in the notebook)
   DATASET_ROOT = "/path/to/your/dataset"
   dataset_model = ImageFolder(root=DATASET_ROOT, transform=model_transform)
   dataset_fm    = ImageFolder(root=DATASET_ROOT, transform=fm_transform)

   # 4) Set up SemanticLens components
   layer_to_analyze = "layer4"
   cache_dir = "./semanticlens_cache"

   cv = ActivationComponentVisualizer(
       model=model,
       dataset_model=dataset_model,
       dataset_fm=dataset_fm,
       layer_names=[layer_to_analyze],
       num_samples=20,  # top-activating images per neuron
       aggregate_fn=aggregators.aggregate_conv_mean,
       cache_dir=cache_dir,
   )
   fm = ClipMobile(device=device)
   lens = sl.Lens(fm, device=device)

   # 5) Run the pipeline (results are cached)
   cv.run(batch_size=64)
   concept_db = lens.compute_concept_db(cv, batch_size=64)

   # Now you can query, score clarity, or visualize components via `cv`/`lens`.


Next steps
----------
- Open the **Quickstart notebook** for detailed explanations, alternatives (datasets, layers, batching) and visual output.
- See the **API reference** for all parameters (`ActivationComponentVisualizer`, `ClipMobile`, `Lens`, aggregators, etc.).
