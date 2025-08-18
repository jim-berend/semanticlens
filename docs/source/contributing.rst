Contributing to SemanticLens
============================

We welcome contributions to SemanticLens! This guide will help you get started.

Development Setup
-----------------

1. **Fork and clone the repository:**

   .. code-block:: bash

      git clone https://github.com/yourusername/semanticlens.git
      cd semanticlens

2. **Install in development mode:**

   .. code-block:: bash

      uv sync --dev

3. **Install pre-commit hooks (optional but recommended):**

   .. code-block:: bash

      pre-commit install

Code Style
----------

SemanticLens uses several tools to maintain code quality:

- **Ruff** for linting and formatting
- **pytest** for testing
- **NumPy style docstrings** for documentation

Run the linter before submitting:

.. code-block:: bash

   ruff check .
   ruff format .

Testing
-------

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=semanticlens

   # Run specific test file
   pytest tests/test_lens.py

Writing Tests
~~~~~~~~~~~~~

When adding new features, please include tests:

.. code-block:: python

   # tests/test_new_feature.py
   import pytest
   from semanticlens import YourNewClass

   def test_your_new_feature():
       instance = YourNewClass()
       result = instance.your_method()
       assert result == expected_value

Documentation
-------------

Documentation is built with Sphinx. To build docs locally:

.. code-block:: bash

   cd docs
   make html
   
   # View docs
   open build/html/index.html

**Docstring Guidelines:**

- Use NumPy style docstrings
- Include parameter types and descriptions
- Add examples for public APIs
- Document return values and exceptions

.. code-block:: python

   def your_function(param1: str, param2: int = 10) -> bool:
       """Brief description of the function.
       
       Longer description with more details about what the function
       does and how to use it.
       
       Parameters
       ----------
       param1 : str
           Description of the first parameter.
       param2 : int, optional
           Description of the second parameter (default is 10).
           
       Returns
       -------
       bool
           Description of what is returned.
           
       Examples
       --------
       >>> result = your_function("example", 5)
       >>> print(result)
       True
       """
       return True

Submitting Changes
------------------

1. **Create a new branch for your feature:**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes and commit:**

   .. code-block:: bash

      git add .
      git commit -m "Add your descriptive commit message"

3. **Push to your fork:**

   .. code-block:: bash

      git push origin feature/your-feature-name

4. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

Types of Contributions
----------------------

We welcome several types of contributions:

**Bug Fixes**
   Found a bug? Please report it in the GitHub issues or submit a fix.

**New Features**
   Have an idea for a new component visualizer or foundation model integration?
   Open an issue to discuss it first.

**Documentation**
   Improvements to docs, tutorials, or examples are always appreciated.

**Performance Improvements**
   Optimizations to existing code are welcome.

**Tests**
   Additional test coverage helps ensure reliability.

Code Review Process
-------------------

All submissions go through code review:

1. Automated checks (tests, linting) must pass
2. At least one maintainer will review your code
3. You may be asked to make changes
4. Once approved, your code will be merged

Guidelines for Good PRs
-----------------------

- **Keep changes focused** - one feature/fix per PR
- **Write descriptive commit messages**
- **Add tests** for new functionality
- **Update documentation** if needed
- **Follow the existing code style**
- **Be responsive** to review feedback

Reporting Issues
----------------

When reporting bugs or requesting features:

1. **Check existing issues** first
2. **Use the issue templates** if available
3. **Provide minimal reproducible examples**
4. **Include environment details** (Python version, OS, etc.)

Community Guidelines
--------------------

- Be respectful and inclusive
- Help others learn and contribute
- Focus on constructive feedback
- Follow the project's code of conduct

Getting Help
------------

If you need help:

- Open a GitHub issue for bugs or feature requests
- Check the documentation and tutorials
- Look at existing code for examples
- Ask questions in discussions

Thank you for contributing to SemanticLens! 🔍
