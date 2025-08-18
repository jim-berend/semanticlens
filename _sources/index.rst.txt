SemanticLens Documentation
==========================

.. raw:: html

   <div align="center">
   <img src="_static/logo-with-name_big.svg" width="400px" alt="SemanticLens logo" align="center" />
   <p>
   An open-source PyTorch library for interpreting and validating large vision models.
   <br>
   Read the paper now as part of <a href="https://www.nature.com/articles/s42256-025-01084-w">Nature Machine Intelligence</a> (Open Access).
   </p>
   </div>

.. raw:: html

   <div align="center" style="margin: 2rem 0;">
     <a href="https://www.nature.com/articles/s42256-025-01084-w">
       <img src="https://img.shields.io/static/v1?label=Nature&message=Machine%20Intelligence&color=green">
     </a>
     <a href="https://doi.org/10.5281/zenodo.15233581">
       <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.15233581.svg">
     </a>
     <img src="https://img.shields.io/badge/Python-3.9, 3.10, 3.11-efefef">
     <a href="https://github.com/jim-berend/semanticlens/blob/main/LICENSE">
       <img alt="License" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg">
     </a>
   </div>

Overview
--------

**SemanticLens** is a universal framework for explaining and validating large vision models. 
While deep learning models are powerful, their internal workings are often a "black box," 
making them difficult to trust and debug. SemanticLens addresses this by mapping the 
internal components of a model (like neurons or filters) into the rich, semantic space 
of a foundation model (e.g., CLIP or SigLIP).

This allows you to "translate" what the model is doing into a human-understandable format, 
enabling you to search, analyze, and audit its internal representations.

Key Features
------------

🔍 **Component Analysis**
   Identify and visualize what individual neurons and layers have learned

📚 **Text Probing** 
   Search model internals using natural language queries

🌄 **Image Probing** 
   Search model internals using natural image queries

📊 **Quantitative Metrics**
   Measure clarity, polysemanticity, and redundancy of learned concepts

🧠 **Foundation Model Integration**
   Built-in support for CLIP, SigLIP, and other vision-language models

🎯 **Multiple Visualization Strategies**
   From activation maximization to attribution-based analysis

Quick Start
-----------

Install SemanticLens:

.. code-block:: bash

   pip install semanticlens

Basic usage:

.. code-block:: python

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

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:
   
   quickstart
   tutorials


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules


.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:
   
   contributing
   license

Citation
--------

If you use SemanticLens in your research, please cite our paper:

.. code-block:: bibtex

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



