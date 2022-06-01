#!/usr/bin/env python
import glob
import os
import re
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from typing import List, Tuple

from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_SOURCE = os.path.join(_PATH_ROOT, "src")
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Load requirements from a file.

    >>> _load_requirements(_PATH_ROOT)
    ['numpy...', 'torch..."]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            char_idx = min(ln.index(ch) for ch in comment_char)
            ln = ln[:char_idx].strip()
        # skip directly installed dependencies
        if ln.startswith("http") or ln.startswith("git") or ln.startswith("-r") or "@" in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion.

    >>> _load_readme_description(_PATH_ROOT, "",  "")
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fp:
        text = fp.read()

    # https://github.com/PyTorchLightning/torchmetrics/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we replace some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={version}")
    text = text.replace("torchmetrics.readthedocs.io/en/stable/", f"torchmetrics.readthedocs.io/en/{version}")
    # codecov badge
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{version}/graph/badge.svg")
    # replace github badges for release ones
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={version}")
    # Azure...
    text = text.replace("?branchName=master", f"?branchName=refs%2Ftags%2F{version}")
    text = re.sub(r"\?definitionId=\d+&branchName=master", f"?definitionId=2&branchName=refs%2Ftags%2F{version}", text)

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    # todo: wrap content as commented description
    text = re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)

    return text


def _load_py_module(fname, pkg="torchmetrics"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_SOURCE, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


ABOUT = _load_py_module("__about__.py")
LONG_DESCRIPTION = _load_readme_description(
    _PATH_ROOT,
    homepage=ABOUT.__homepage__,
    version=f"v{ABOUT.__version__}",
)
BASE_REQUIREMENTS = _load_requirements(path_dir=_PATH_ROOT, file_name="requirements.txt")


def _prepare_extras(skip_files: Tuple[str] = ("devel.txt",)):
    # find all extra requirements
    _load_req = partial(_load_requirements, path_dir=_PATH_REQUIRE)
    found_req_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(_PATH_REQUIRE, "*.txt")))
    # filter unwanted files
    found_req_files = [n for n in found_req_files if n not in skip_files]
    found_req_names = [os.path.splitext(req)[0] for req in found_req_files]
    # define basic and extra extras
    extras_req = {
        name: _load_req(file_name=fname) for name, fname in zip(found_req_names, found_req_files) if "_test" not in name
    }
    for name, fname in zip(found_req_names, found_req_files):
        if "_test" in name:
            extras_req["test"] += _load_req(file_name=fname)
    # filter the uniques
    extras_req = {n: list(set(req)) for n, req in extras_req.items()}
    # create an 'all' keyword that install all possible dependencies
    extras_req["all"] = list(chain([pkgs for k, pkgs in extras_req.items() if k not in ("test", "docs")]))
    return extras_req


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
if __name__ == "__main__":
    setup(
        name="torchmetrics",
        version=ABOUT.__version__,
        description=ABOUT.__docs__,
        author=ABOUT.__author__,
        author_email=ABOUT.__author_email__,
        url=ABOUT.__homepage__,
        download_url=os.path.join(ABOUT.__homepage__, "archive", "master.zip"),
        license=ABOUT.__license__,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        include_package_data=True,
        zip_safe=False,
        keywords=["deep learning", "machine learning", "pytorch", "metrics", "AI"],
        python_requires=">=3.7",
        setup_requires=[],
        install_requires=BASE_REQUIREMENTS,
        extras_require=_prepare_extras(),
        project_urls={
            "Bug Tracker": os.path.join(ABOUT.__homepage__, "issues"),
            "Documentation": "https://torchmetrics.rtfd.io/en/latest/",
            "Source Code": ABOUT.__homepage__,
        },
        classifiers=[
            "Environment :: Console",
            "Natural Language :: English",
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            "Development Status :: 5 - Production/Stable",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
    )
