# Configuration file for the Sphinx documentation builder.
import os
import sys

# Path setup
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../semanticlens"))

# -- Project information -----------------------------------------------------
project = "SemanticLens"
copyright = "2025, Jim Berend"
author = "Jim Berend"
release = "0.1.2"
version = "0.1.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    # "nbsphinx",  # For Jupyter notebooks - commented out for now
    # "myst_parser",  # For markdown support - commented out for now
]

# Template paths
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": None,
    # '.md': 'myst_parser',  # Commented out for now
}

# Autosummary configuration
autosummary_generate = True
autosummary_imported_members = True

# Napoleon configuration (NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

# Notebook configuration - commented out for now
# nbsphinx_execute = 'never'  # Don't execute notebooks during build
# nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = f"{project} v{version}"
html_short_title = project

# Theme-specific options
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-small.svg",
    "dark_logo": "logo-small.svg",
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
        "color-admonition-background": "transparent",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jim-berend/semanticlens",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/semanticlens/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/jim-berend/semanticlens/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Static files and favicon
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Additional files to copy
html_extra_path = []

# Logo and favicon
# html_logo = "_static/logo-with-name_big.svg"
# html_favicon = "_static/favicon.ico"

# Show source links
html_show_sourcelink = True
html_copy_source = True
html_show_sphinx = True
html_show_copyright = True

# Last updated format
html_last_updated_fmt = "%b %d, %Y"

# Search language
html_search_language = "en"

# Custom sidebar
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
