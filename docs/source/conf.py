import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

sys.path.insert(0, os.path.abspath("../.."))

# ── Project information ───────────────────────────────────────────────────────
project   = "StarkZee"
copyright = "2026, G. Ronchi"
author    = "G. Ronchi"

# Derived from the installed package (setuptools_scm, driven by git tags) so
# this can't silently drift out of sync with the actual release, as the
# previous hardcoded "0.1.0" had.
try:
    release = _pkg_version("starkzee")
except PackageNotFoundError:
    release = "0.0.0"
version = release.split("+")[0]  # short X.Y.Z, dropping the local dev/commit suffix

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
# Pinned to MathJax v4 explicitly rather than relying on sphinx.ext.mathjax's
# own default CDN URL, so rendering doesn't silently break again if a future
# Sphinx release moves that default to v5+. GitHub Pages viewers have internet
# access, so a pinned CDN URL (rather than vendoring the file) is sufficient.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"

# Fix span.eqno positioning: MathJax renders display math as a block-level
# mjx-container, which breaks Sphinx's float:right approach.
#
# This is injected as a raw inline <script> (via setup() below) rather than
# through Sphinx's mathjax3_config/mathjax4_config config values: those are
# always passed through json.dumps(), which turns a Python string containing
# JS source into a *JS string literal*, not a callable. MathJax v3 tolerated
# that (some internal eval/coercion made "ready" work as a string); MathJax
# v4 does a strict typeof check and throws
# "MathJax.config.startup.ready is not a function", silently aborting
# MathJax entirely -- confirmed by loading the real CDN script in a headless
# browser with each form. Defining window.MathJax directly in raw JS (a
# genuine function, not a JSON-serialized string) works under both v3 and v4
# and isn't tied to whichever mathjaxN_config key a given Sphinx version
# happens to support.
_MATHJAX_READY_JS = """
window.MathJax = {
    startup: {
        ready: () => {
            MathJax.startup.defaultReady();
            const style = document.createElement('style');
            style.textContent = [
                'div.math { position: relative; padding-right: 5em; }',
                'span.eqno { float: none; position: absolute; right: 0;',
                '            top: 50%; transform: translateY(-50%); }'
            ].join('\\n');
            document.head.appendChild(style);
        }
    }
};
"""


def setup(app):
    # Priority below sphinx.ext.mathjax's default (500) so this config
    # script is emitted, and therefore executes, before the mathjax_path
    # loader script tag -- window.MathJax must exist before that script runs.
    app.add_js_file(None, priority=100, body=_MATHJAX_READY_JS)

# ── Misc ──────────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path   = []

# Suppress cross-reference warnings for bare type names (ndarray, array-like, etc.)
# that appear in docstrings but aren't fully qualified Sphinx targets.
suppress_warnings = ["ref.class", "ref.any", "ref.func"]
