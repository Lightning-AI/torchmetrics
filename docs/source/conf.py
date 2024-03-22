#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import glob
import inspect
import os
import re
import shutil
import sys
from typing import Optional

import lai_sphinx_theme
import torchmetrics
from lightning_utilities.docs.formatting import _transform_changelog

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.realpath(os.path.join(_PATH_HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(_PATH_ROOT))

FOLDER_GENERATED = "generated"
SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))
SPHINX_FETCH_ASSETS = int(os.environ.get("SPHINX_FETCH_ASSETS", False))
SPHINX_PIN_RELEASE_VERSIONS = int(os.getenv("SPHINX_PIN_RELEASE_VERSIONS", False))

html_favicon = "_static/images/icon.svg"

# -- Project information -----------------------------------------------------

# this name shall match the project name in Github as it is used for linking to code
project = "PyTorch-Metrics"
copyright = torchmetrics.__copyright__
author = torchmetrics.__author__

# The short X.Y version
version = torchmetrics.__version__
# The full version, including alpha/beta/rc tags
release = torchmetrics.__version__

# Options for the linkcode extension
# ----------------------------------
github_user = "Lightning-AI"
github_repo = "metrics"

# -- Project documents -------------------------------------------------------


os.makedirs(os.path.join(_PATH_HERE, FOLDER_GENERATED), exist_ok=True)
# copy all documents from GH templates like contribution guide
for md in glob.glob(os.path.join(_PATH_ROOT, ".github", "*.md")):
    shutil.copy(md, os.path.join(_PATH_HERE, FOLDER_GENERATED, os.path.basename(md)))
# copy also the changelog
_transform_changelog(
    os.path.join(_PATH_ROOT, "CHANGELOG.md"),
    os.path.join(_PATH_HERE, FOLDER_GENERATED, "CHANGELOG.md"),
)


def _set_root_image_path(page_path: str) -> None:
    """Set relative path to be from the root, drop all `../` in images used gallery."""
    with open(page_path, encoding="UTF-8") as fopen:
        body = fopen.read()
    found = re.findall(r"   :image: (.*)\.svg", body)
    for occur in found:
        occur_ = occur.replace("../", "")
        body = body.replace(occur, occur_)
    with open(page_path, "w", encoding="UTF-8") as fopen:
        fopen.write(body)


if SPHINX_FETCH_ASSETS:
    from lightning_utilities.docs import fetch_external_assets

    fetch_external_assets(
        docs_folder=_PATH_HERE,
        assets_folder="_static/fetched-s3-assets",
        retrieve_pattern=r"https?://[-a-zA-Z0-9_]+\.s3\.[-a-zA-Z0-9()_\\+.\\/=]+",
    )
    all_pages = glob.glob(os.path.join(_PATH_HERE, "**", "*.rst"), recursive=True)
    for page in all_pages:
        _set_root_image_path(page)


if SPHINX_PIN_RELEASE_VERSIONS:
    from lightning_utilities.docs import adjust_linked_external_docs

    adjust_linked_external_docs(
        "https://numpy.org/doc/stable/", "https://numpy.org/doc/{numpy.__version__}/", _PATH_ROOT
    )
    adjust_linked_external_docs(
        "https://pytorch.org/docs/stable/", "https://pytorch.org/docs/{torch.__version__}/", _PATH_ROOT
    )
    adjust_linked_external_docs(
        "https://matplotlib.org/stable/",
        "https://matplotlib.org/{matplotlib.__version__}/",
        _PATH_ROOT,
        version_digits=3,
    )
    adjust_linked_external_docs(
        "https://scikit-learn.org/stable/", "https://scikit-learn.org/{sklearn.__version__}/", _PATH_ROOT
    )


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

needs_sphinx = "5.3"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_paramlinks",
    "sphinx.ext.githubpages",
    "lai_sphinx_theme.extensions.lightning",
    "matplotlib.sphinxext.plot_directive",
]

# Set that source code from plotting is always included
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# https://berkeley-stat159-f17.github.io/stat159-f17/lectures/14-sphinx..html#conf.py-(cont.)
# https://stackoverflow.com/questions/38526888/embed-ipython-notebook-in-sphinx-document
# I execute the notebooks manually in advance. If notebooks test the code,
# they should be run at build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_requirejs_path = ""

myst_update_mathjax = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    os.path.join(FOLDER_GENERATED, "PULL_REQUEST_TEMPLATE.md"),
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "lai_sphinx_theme"
html_theme_path = [lai_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "https://pytorchlightning.ai",
    "canonical_url": torchmetrics.__docs_url__,
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

html_logo = "_static/images/logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + "-doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project + ".tex", project + " Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, project, project + " Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        project + " Documentation",
        author,
        project,
        torchmetrics.__docs__,
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
nitpicky = True

# -- Options for to-do extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# https://github.com/rtfd/readthedocs.org/issues/1139
# I use sphinx-apidoc to auto-generate API documentation for my project.
# Right now I have to commit these auto-generated files to my repository
# so that RTD can build them into HTML docs. It'd be cool if RTD could run
# sphinx-apidoc for me, since it's easy to forget to regen API docs
# and commit them to my repo after making changes to my code.

# packages for which sphinx-apidoc should generate the docs (.rst files)
PACKAGES = [
    torchmetrics.__name__,
]


def setup(app) -> None:  # noqa: ANN001
    # this is for hiding doctest decoration,
    # see: http://z4r.github.io/python/2011/12/02/hides-the-prompts-and-output/
    app.add_js_file("copybutton.js")
    # app.connect('builder-inited', run_apidoc)


# copy all notebooks to local folder
path_nbs = os.path.join(_PATH_HERE, "notebooks")
os.makedirs(path_nbs, exist_ok=True)
for path_ipynb in glob.glob(os.path.join(_PATH_ROOT, "notebooks", "*.ipynb")):
    path_ipynb2 = os.path.join(path_nbs, os.path.basename(path_ipynb))
    shutil.copy(path_ipynb, path_ipynb2)


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def package_list_from_file(file: str) -> list[str]:
    mocked_packages = []
    with open(file) as fp:
        for ln in fp.readlines():
            found = [ln.index(ch) for ch in list(",=<>#") if ch in ln]
            pkg = ln[: min(found)] if found else ln
            if pkg.rstrip():
                mocked_packages.append(pkg.rstrip())
    return mocked_packages


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    "PyYAML": "yaml",
}
MOCK_PACKAGES = []
if SPHINX_MOCK_REQUIREMENTS:
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += package_list_from_file(os.path.join(_PATH_ROOT, "requirements", "_docs.txt"))
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]

autodoc_mock_imports = MOCK_PACKAGES


# Resolve function
# This function is used to populate the (source) links in the API
def linkcode_resolve(domain, info) -> Optional[str]:  # noqa: ANN001
    # try to find the file and line number, based on code from numpy:
    # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L424

    if domain != "py" or not info["module"]:
        return None

    obj = _get_obj(info)
    file_name = _get_file_name(obj)

    if not file_name:
        return None

    line_str = _get_line_str(obj)
    version_str = _get_version_str()

    return f"https://github.com/{github_user}/{github_repo}/blob/{version_str}/src/torchmetrics/{file_name}{line_str}"


def _get_obj(info: dict) -> object:
    module_name = info["module"]
    full_name = info["fullname"]
    sub_module = sys.modules.get(module_name)
    obj = sub_module
    for part in full_name.split("."):
        obj = getattr(obj, part)
    # strip decorators, which would resolve to the source of the decorator
    return inspect.unwrap(obj)


def _get_file_name(obj) -> str:  # noqa: ANN001
    try:
        file_name = inspect.getsourcefile(obj)
        file_name = os.path.relpath(file_name, start=os.path.dirname(torchmetrics.__file__))
    except TypeError:  # This seems to happen when obj is a property
        file_name = None
    return file_name


def _get_line_str(obj) -> str:  # noqa: ANN001
    source, line_number = inspect.getsourcelines(obj)
    return "#L%d-L%d" % (line_number, line_number + len(source) - 1)


def _get_version_str() -> str:
    if any(s in torchmetrics.__version__ for s in ("dev", "rc")):
        version_str = "master"
    else:
        version_str = f"v{torchmetrics.__version__}"
    return version_str


autosummary_generate = True

autodoc_member_order = "groupwise"

autoclass_content = "class"

autodoc_default_options = {
    "members": True,
    # 'methods': True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    # 'show-inheritance': True,
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True

# only run doctests marked with a ".. doctest::" directive
doctest_test_doctest_blocks = ""
doctest_global_setup = """

import os
import torch

from torch import Tensor
from torchmetrics import Metric

"""
coverage_skip_undoc_in_source = True

# skip false positive linkcheck errors from anchors
linkcheck_anchors = False

# A timeout value, in seconds, for the linkcheck builder.
linkcheck_timeout = 30

# ignore all links in any CHANGELOG file
linkcheck_exclude_documents = [r"^(.*\/)*CHANGELOG.*$"]

# jstor and sciencedirect cannot be accessed from python, but links work fine in a local doc
linkcheck_ignore = [
    # The Treatment of Ties in Ranking Problems
    "https://www.jstor.org/stable/2332303",
    # Quality Assessment of Deblocked Images
    "https://ieeexplore.ieee.org/abstract/document/5535179",
    # Image information and visual quality
    "https://ieeexplore.ieee.org/abstract/document/1576816",
    # Performance measurement in blind audio source separation
    "https://ieeexplore.ieee.org/abstract/document/1643671",
    # A Non-Intrusive Quality and Intelligibility Measure of Reverberant and Dereverberated Speech
    "https://ieeexplore.ieee.org/abstract/document/5547575",
    # An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers
    "https://ieeexplore.ieee.org/abstract/document/7539284",
    # A short-time objective intelligibility measure for time-frequency weighted noisy speech
    "https://ieeexplore.ieee.org/abstract/document/5495701",
    # An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech
    "https://ieeexplore.ieee.org/abstract/document/5713237",
    # A universal image quality index
    "https://ieeexplore.ieee.org/abstract/document/995823",
    # On the Performance Evaluation of Pan-Sharpening Techniques
    "https://ieeexplore.ieee.org/abstract/document/4317530",
    # Robust parameter estimation with a small bias against heavy contamination
    "https://www.sciencedirect.com/science/article/pii/S0047259X08000456",
]
