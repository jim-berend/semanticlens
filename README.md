<div align="center">
  <img src="https://github.com/jim-berend/semanticlens/blob/be718f96ba7c52b29249ff7b4806999890895c72/static/images/logo-with-name_big.svg" width="400px" alt="SemanticLens logo" align="center" />
  <p>
  An open-source PyTorch library for interpreting and validating large vision models.
  <br>
  Read the paper now as part of <a href="https://www.nature.com/articles/s42256-025-01084-w">Nature Machine Intelligence</a> (Open Access).
  </p>
</div>

<br>

<div align="center">
  <a href="https://www.nature.com/articles/s42256-025-01084-w">
    <img  src="https://img.shields.io/static/v1?label=Nature&message=Machine%20Intelligence&color=green">
  </a>
  <a href="https://doi.org/10.5281/zenodo.15233581">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.15233581.svg">
  </a>
  <a href="https://pypi.org/project/semanticlens/">
    <img alt="pypi" src="https://img.shields.io/pypi/v/semanticlens">
  </a>
  <img  src="https://img.shields.io/badge/Python-3.9, 3.10, 3.11-efefef">
  <a href="LICENSE">
    <img alt="DOI" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg">
  </a>
	<img alt="PyLint" src="https://github.com/jim-berend/semanticlens/actions/workflows/ruff-lint.yml/badge.svg">
  <a href="https://jim-berend.github.io/semanticlens/">
    <img  src="https://img.shields.io/badge/Docs-SemanticLens-ff8c00">
  </a>
</div>


**SemanticLens** is a universal framework for explaining and validating large vision models. While deep learning models are powerful, their internal workings are often a "black box," making them difficult to trust and debug. SemanticLens addresses this by mapping the internal components of a model (like neurons or filters) into the rich, semantic space of a foundation model (e.g., CLIP or SigLIP).

This allows you to "translate" what the model is doing into a human-understandable format, enabling you to search, analyze, and audit its internal representations.


## How It Works


<div align="center">
  <img src="https://github.com/jim-berend/semanticlens/blob/be718f96ba7c52b29249ff7b4806999890895c72/static/images/overview-figure.svg" width="90%" alt="Overview figure" align="center" />
  <p>
  Overview of the SemanticLens framework as introduced in our <a href="https://www.nature.com/articles/s42256-025-01084-w"> research paper.</a>

  </p>
</div>

The core workflow of SemanticLens involves three main steps:
1) **Collect**: For each component in a model M, we identify the data samples that cause the highest activation (the "concept examples").
We provide a suite of [`ComponentVisualizers`](semanticlens/component_visualization) that implement different strategies, from simple activation maximization to relevance-maximization and attribution-based cropping.

2) **Embed**: These examples are then fed into a foundation model (like CLIP), which creates a meaningful vector representation for each component. SemanticLens includes built-in support for [OpenCLIP](https://github.com/mlfoundations/open_clip) and can be easily extended with other foundation models (see [base.py](semanticlens/foundation_models/base.py)).


3) **Analyze**: These vector representations enable powerful analyses. The [`Lens`](semanticlens/lens.py) class is the main interface for this, orchestrating the preprocessing, caching, and evaluation needed to search and audit your model using its new semantic embeddings.


## Installation

You can install SemanticLens directly from PyPI:
```bash
pip install semanticlens
```

To install the latest version from this repository:

```bash
pip install git+https://github.com/jim-berend/semanticlens.git
```

## Quickstart
Example usage:
```python
import semanticlens as sl

... # dataset and model setup

# Initialization

cv = sl.component_visualization.ActivationComponentVisualizer(
    model,
    dataset_model,
    dataset_fm,
    layer_names=layer_names,
    device=device,
    cache_dir=cache_dir,
)

fm = sl.foundation_models.OpenClip(url="RN50", pretrained="openai", device=device)

lens = sl.Lens(fm, device=device)

# Semantic Embedding 

concept_db = lens.compute_concept_db(cv, batch_size=128, num_workers=8)
aggregated_cpt_db = {k: v.mean(1) for k, v in concept_db.items()}

# Analysis

polysemanticity_scores = lens.eval_polysemanticity(concept_db)

search_results = lens.text_probing(["cats", "dogs"], aggregated_cpt_db)

...
```
<a href="tutorials/quickstart.ipynb">
<img  src="https://img.shields.io/badge/Tutorial-Quickstart.ipynb-2881db">
</a>

Full quickstart guide: [quickstart.ipynb](tutorials/quickstart.ipynb)


<a href="https://jim-berend.github.io/semanticlens/">
<img  src="https://img.shields.io/badge/Docs-SemanticLens-ff8c00">
</a>

Package documentation: [docs](https://jim-berend.github.io/semanticlens/) 

## Contributing

We welcome contributions to SemanticLens! Whether you're fixing a bug, adding a new feature, or improving the documentation, your help is appreciated. 

If you'd like to contribute, please follow these steps:
1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix (git checkout -b feature/your-feature-name).
3. Make your changes and commit them with a clear message.
4. Open a pull request to the main branch of the original repository.

For bug reports or feature requests, please use the GitHub Issues section. Before starting work on a major change, it's a good idea to open an issue first to discuss your plan.

## License

[BSD 3-Clause License](LICENSE)


## Citation
```
@article{dreyer_mechanistic_2025,
	title = {Mechanistic understanding and validation of large {AI} models with {SemanticLens}},
	copyright = {2025 The Author(s)},
	issn = {2522-5839},
	url = {https://www.nature.com/articles/s42256-025-01084-w},
	doi = {10.1038/s42256-025-01084-w},
	language = {en},
	urldate = {2025-08-18},
	journal = {Nature Machine Intelligence},
	author = {Dreyer, Maximilian and Berend, Jim and Labarta, Tobias and Vielhaben, Johanna and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
	month = aug,
	year = {2025},
	note = {Publisher: Nature Publishing Group},
	keywords = {Computer science, Information technology},
	pages = {1--14},
}
```

