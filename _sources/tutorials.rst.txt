Tutorials
=========

Interactive tutorials and examples for SemanticLens.

Getting Started
---------------

The following tutorial provides hands-on examples of using SemanticLens. 
You can find the interactive Jupyter notebook in the ``tutorials/`` 
directory of the repository.

**Available Tutorial:**

- **Quickstart Tutorial** (``quickstart.ipynb``): A comprehensive introduction to SemanticLens covering the core workflow from setup to text probing, interpretability metrics, and advanced usage patterns. Perfect for getting started with the library.

.. note:: 
   Interactive notebook rendering will be added in a future documentation update.
   For now, please refer to the notebook files in the repository.

Tutorial Contents
-----------------

**Quickstart Tutorial**
   A basic introduction to SemanticLens covering the core workflow from setup to text probing.
   Perfect for first-time users.


Running the Tutorials
----------------------

To run these tutorials interactively:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/jim-berend/semanticlens.git
      cd semanticlens

2. Install dependencies:

   .. code-block:: bash

      uv sync
      uv pip install jupyter

3. Start Jupyter:

   .. code-block:: bash

      jupyter notebook tutorials/

4. Open any of the notebook files (.ipynb) to run them interactively.

Additional Examples
-------------------

For more examples and use cases, check out:

- The ``tutorials/`` directory in the repository 
- Test files in ``tests/`` for API usage patterns
