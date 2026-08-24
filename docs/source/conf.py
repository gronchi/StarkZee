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
    "sphinx_rtd_theme",          # Read the Docs HTML theme
    "sphinxcontrib.mermaid",     # maintainable workflow and decision diagrams
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
}

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme   = "sphinx_rtd_theme"
html_title   = "StarkZee"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}

# ── MathJax ───────────────────────────────────────────────────────────────────
# Self-hosted (docs/source/_static/mathjax/tex-mml-chtml.js) instead of the
# sphinx.ext.mathjax default CDN URL, so equations render without requiring
# internet access when the built HTML is viewed (e.g. on an air-gapped or
# firewalled machine). Re-download from
# https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js to update.
mathjax_path = "mathjax/tex-mml-chtml.js"

# Fix span.eqno positioning: MathJax 3 renders display math as a block-level
# mjx-container, which breaks Sphinx's float:right approach.
mathjax3_config = {
    "startup": {
        "ready": """() => {
            MathJax.startup.defaultReady();
            const style = document.createElement('style');
            style.textContent = [
                'div.math { position: relative; padding-right: 5em; }',
                'span.eqno { float: none; position: absolute; right: 0;',
                '            top: 50%; transform: translateY(-50%); }'
            ].join('\\n');
            document.head.appendChild(style);
        }"""
    }
}

# ── Misc ──────────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path   = []

# Suppress cross-reference warnings for bare type names (ndarray, array-like, etc.)
# that appear in docstrings but aren't fully qualified Sphinx targets.
suppress_warnings = ["ref.class", "ref.any", "ref.func"]
