#!/usr/bin/env python
import glob
import os
import re
from collections.abc import Iterable, Iterator
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from pathlib import Path
from typing import Any, Optional, Union

from pkg_resources import Requirement, yield_lines
from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_SOURCE = os.path.join(_PATH_ROOT, "src")
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")
_FREEZE_REQUIREMENTS = os.environ.get("FREEZE_REQUIREMENTS", "0").lower() in ("1", "true")


class _RequirementWithComment(Requirement):
    strict_string = "# strict"

    def __init__(self, *args: Any, comment: str = "", pip_argument: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        if pip_argument is not None and not pip_argument:
            raise ValueError("Expected `pip_argument` to either be `None` or an str, but got an empty string")
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: bool) -> str:
        """Remove version restrictions unless they are strict.

        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# anything").adjust(False)
        'arrow<=1.2.2,>=1.2.0'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# strict").adjust(False)
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# my name").adjust(True)
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust(True)
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow").adjust(True)
        'arrow'

        """
        out = str(self)
        if self.strict:
            return f"{out}  {self.strict_string}"
        if unfreeze:
            for operator, version in self.specs:
                if operator in ("<", "<="):
                    # drop upper bound
                    return out.replace(f"{operator}{version},", "")
        return out


def _parse_requirements(strs: Union[str, Iterable[str]]) -> Iterator[_RequirementWithComment]:
    r"""Adapted from `pkg_resources.parse_requirements` to include comments.

    >>> txt = ['# ignored', '', 'this # is an', '--piparg', 'example', 'foo # strict', 'thing', '-r different/file.txt']
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']
    >>> txt = '\\n'.join(txt)
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']

    """
    lines = yield_lines(strs)
    pip_argument = None
    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            comment_pos = line.find(" #")
            line, comment = line[:comment_pos], line[comment_pos:]
        else:
            comment = ""
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith("\\"):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        if "@" in line or re.search("https?://", line):
            # skip lines with links like `pesq @ git+https://github.com/ludlows/python-pesq`
            continue
        # If there's a pip argument, save it
        if line.startswith("--"):
            pip_argument = line
            continue
        if line.startswith("-r "):
            # linked requirement files are unsupported
            continue
        yield _RequirementWithComment(line, comment=comment, pip_argument=pip_argument)
        pip_argument = None


def _load_requirements(
    path_dir: str, file_name: str = "base.txt", unfreeze: bool = not _FREEZE_REQUIREMENTS
) -> list[str]:
    """Load requirements from a file.

    >>> _load_requirements(_PATH_REQUIRE)
    ['numpy...', 'torch..."]

    """
    path = Path(path_dir) / file_name
    if not path.exists():
        raise ValueError("Path {path} not found for input dir {path_dir} and filename {file_name}.")
    text = path.read_text()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion.

    >>> _load_readme_description(_PATH_ROOT, "",  "")
    '<div align="center">...'

    """
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fp:
        text = fp.read()

    # https://github.com/Lightning-AI/torchmetrics/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
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
    return re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)


def _load_py_module(fname: str, pkg: str = "torchmetrics"):
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
BASE_REQUIREMENTS = _load_requirements(path_dir=_PATH_REQUIRE, file_name="base.txt")


def _prepare_extras(skip_pattern: str = "^_", skip_files: tuple[str] = ("base.txt",)) -> dict:
    """Preparing extras for the package listing requirements.

    Args:
        skip_pattern: ignore files with this pattern, by default all files starting with _
        skip_files: ignore some additional files, by default base requirements

    Note, particular domain test requirement are aggregated in single "_tests" extra (which is not accessible).

    """
    # find all extra requirements
    _load_req = partial(_load_requirements, path_dir=_PATH_REQUIRE)
    found_req_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(_PATH_REQUIRE, "*.txt")))
    # filter unwanted files
    found_req_files = [n for n in found_req_files if not re.match(skip_pattern, n)]
    found_req_files = [n for n in found_req_files if n not in skip_files]
    found_req_names = [os.path.splitext(req)[0] for req in found_req_files]
    # define basic and extra extras
    extras_req = {"_tests": []}
    for name, fname in zip(found_req_names, found_req_files):
        if name.endswith("_test"):
            extras_req["_tests"] += _load_req(file_name=fname)
        else:
            extras_req[name] = _load_req(file_name=fname)
    # filter the uniques
    extras_req = {n: list(set(req)) for n, req in extras_req.items()}
    # create an 'all' keyword that install all possible dependencies
    extras_req["all"] = list(chain([pkgs for k, pkgs in extras_req.items() if k not in ("_test", "_tests")]))
    extras_req["dev"] = extras_req["all"] + extras_req["_tests"]
    return {k: v for k, v in extras_req.items() if not k.startswith("_")}


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
        python_requires=">=3.9",
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
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )
