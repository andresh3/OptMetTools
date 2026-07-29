# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OptMetTools'
copyright = '2026, Andrew Humphreys'
author = 'Andrew Humphreys'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.mathjax",
        "myst_parser"
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

napoleon_google_docstring = False  # Turn off Google style if you only want NumPy
napoleon_numpy_docstring = True    # Explicitly ensure NumPy style is enabled
napoleon_use_param = False         # Keeps the NumPy style parameter list

latex_elements = {
    # For single-sided output, which removes most blank pages
    "classoptions": ",oneside",
}
