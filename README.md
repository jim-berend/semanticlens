<div align="center">
  <img src="static/images/logo-with-name_big.svg" width="350"/>
  <p>An open-source pytorch implementation of SemanticLens.
  </p>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2501.05398-b31b1b.svg)](https://arxiv.org/abs/2501.05398)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15233581.svg)](https://doi.org/10.5281/zenodo.15233581)



<!-- ![Tests status](https://img.shields.io/badge/...) -->

<!-- ## Overview

...

<div align="center">
  <img src="./static/images/overview.svg" width="1000"/>å
  <p>
  Tools:
  <a href="./test.txt">Search</a>,
  <a href="./test.txt">Describe</a>,
  <a href="./test.txt">Compare</a>,
  <a href="./test.txt">Audit</a>,
  <a href="./test.txt">Assess Interpretability</a>
  </p>
</div>



Examples -->

## Installation
```bash
pip install semanticlens
```

## Quickstart

```python
import semanticlens as sl

# step 1 - collect component-examples

act_cv = sl.component_visualization.ActivationComponentVisualizer(
    model,
    dataset,
    layer_names,
)
act_cv.run(batch_size=128, num_workers=16)

# step 2 - compute semantic embeddings

fm = sl.foundation_models.OpenClip("hf-hub:apple/MobileCLIP-S2-OpenCLIP")
lens = sl.Lens(
    dataset=dataset,
    component_visualizer=act_cv,
    foundation_model=fm,
    dataset_name="...", # e.g. "imagenet"
    storage_dir="...", # e.g. "cache"
)
lens.compute_semantic_embeddigs(layer_names)

# step 3 - inspect, search, label, evaluate, ...

lens.label(
  ["wall", "small", "sky", "floor", "red",...],
  templates=["a natural image showing {}"],
) # ("layer3", 95): "brown vegetation" ,...

lens.search("watermark") # ("layer4", [63, 21, 362]), ...

lens.eval_clarity() # {"layer4" : [[0.64, 0.32, ... ]]}

```

## Project status

> **Note**  
> The project is currently under active development.  🛠️  
> Please expect interfaces to change.
 
The state of development will be updated here.



## Contributing

We adhere to the [PEP8](https://www.python.org/dev/peps/pep-0008) standard with a maximum line width of 120 characters.  
For linting and style checks, we use `ruff`, configured to enforce PEP8 compliance along with additional rules.  
Our basic tests are implemented using `pytest`.


## License

BSD 3-Clause License


### Citation
```
@article{dreyer2025mechanistic,
  title={Mechanistic understanding and validation of large AI models with SemanticLens},
  author={Dreyer, Maximilian and Berend, Jim and Labarta, Tobias and Vielhaben, Johanna and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  journal={arXiv preprint arXiv:2501.05398},
  year={2025}
}
```
