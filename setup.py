#!/usr/bin/env python
import glob
import os
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from typing import Tuple

from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_py_module(fname, pkg="torchmetrics"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")
setup_tools = _load_py_module("setup_tools.py")
long_description = setup_tools._load_readme_description(
    _PATH_ROOT,
    homepage=about.__homepage__,
    version=f"v{about.__version__}",
)


BASE_REQUIREMENTS = setup_tools._load_requirements(path_dir=_PATH_ROOT, file_name="requirements.txt")


def _prepare_extras(skip_files: Tuple[str] = ("devel.txt",)):
    # find all extra requirements
    _load_req = partial(setup_tools._load_requirements, path_dir=_PATH_REQUIRE)
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
    # create an 'all' keyword that install all possible denpendencies
    extras_req["all"] = [pkg for reqs in extras_req.values() for pkg in reqs]
    return extras_req


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="torchmetrics",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url=os.path.join(about.__homepage__, "archive", "master.zip"),
    license=about.__license__,
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "machine learning", "pytorch", "metrics", "AI"],
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=BASE_REQUIREMENTS,
    extras_require=_prepare_extras(),
    project_urls={
        "Bug Tracker": os.path.join(about.__homepage__, "issues"),
        "Documentation": "https://torchmetrics.rtfd.io/en/latest/",
        "Source Code": about.__homepage__,
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
