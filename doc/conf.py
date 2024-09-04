# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# Add paths to the code to the sphinx system path here
# Assume that the root of the python code is one level up from this file location
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2] / 'beam-python'))
sys.path.insert(0, str(Path(__file__).parents[1] / 'mne/lib64/python3.12/site-packages/'))
sys.path.insert(0, str(Path(__file__).parents[1] / 'mne/lib/python3.12/site-packages/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'religion'
copyright = '2024, amoiseev'
author = 'amoiseev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  'sphinx.ext.autodoc','sphinx.ext.autosummary', 
                'myst_parser',              # To use .md files as sources
                'sphinx.ext.napoleon',      # To use google or numpy doc strings instead of original
                                            # sphinx/rst doc strings
    ]

autodoc_default_options = {
        'members': True		# This adds :members: option to automodule, autoclass by default
    }

napoleon_custom_sections = [('Returns', 'params_style')]    # This allows describing multiple return values
								                            # and their types in 'Returns' section

myst_heading_anchors = 3    # To properly convert links to subsections in .md files

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

add_module_names = False    # Do not prepend the module name to function descriptions


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# To generate docs for __init__() functions (skipped by default):
# https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    # Special processing of the __init__():
    app.connect("autodoc-skip-member", skip)

    # Change the width of the generated docs
    # See https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs
    app.add_css_file('custom.css')

