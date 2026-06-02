import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# ── Project information ───────────────────────────────────────────────────────
project   = "StarkZee"
copyright = "2025, G. Ronchi"
author    = "G. Ronchi"
release   = "0.1.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings
    "sphinx.ext.napoleon",       # NumPy / Google docstring styles
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.mathjax",        # LaTeX math in RST
    "sphinx.ext.intersphinx",    # cross-links to NumPy / SciPy
    "sphinx.ext.autosummary",    # summary tables
]

# ── napoleon settings ─────────────────────────────────────────────────────────
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_use_param         = True
napoleon_use_rtype         = True
napoleon_preprocess_types  = True

# ── autodoc settings ─────────────────────────────────────────────────────────
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "member-order":     "bysource",
}
autodoc_typehints = "description"

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme   = "furo"
html_title   = "StarkZee"
html_static_path = []

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# ── Misc ──────────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path   = []

# Suppress cross-reference warnings for bare type names (ndarray, array-like, etc.)
# that appear in docstrings but aren't fully qualified Sphinx targets.
suppress_warnings = ["ref.class", "ref.any"]
