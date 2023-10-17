# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import json
import logging
import os
import re
import sys
import traceback
from typing import List, Optional, Tuple, Union

import fire
import requests
from packaging.version import parse
from pkg_resources import parse_requirements

_REQUEST_TIMEOUT = 10
_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
_PKG_WIDE_SUBPACKAGES = ("utilities", "helpers")
LUT_PYTHON_TORCH = {
    "3.8": "1.4",
    "3.9": "1.7.1",
    "3.10": "1.11",
    "3.11": "1.13",
}
_path = lambda *ds: os.path.join(_PATH_ROOT, *ds)
REQUIREMENTS_FILES = (*glob.glob(_path("requirements", "*.txt")), _path("requirements.txt"))


def request_url(url: str, auth_token: Optional[str] = None) -> Optional[dict]:
    """General request with checking if request limit was reached."""
    auth_header = {"Authorization": f"token {auth_token}"} if auth_token else {}
    try:
        req = requests.get(url, headers=auth_header, timeout=_REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        traceback.print_exc()
        return None
    if req.status_code == 403:
        return None
    return json.loads(req.content.decode(req.encoding))


class AssistantCLI:
    """CLI assistant for local CI."""

    @staticmethod
    def prune_packages(req_file: str, *pkgs: str) -> None:
        """Prune packages from requirement file."""
        with open(req_file) as fp:
            lines = fp.readlines()

        for pkg in pkgs:
            lines = [ln for ln in lines if not ln.startswith(pkg)]
        logging.info(lines)

        with open(req_file, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    @staticmethod
    def set_min_torch_by_python(fpath: str = "requirements/base.txt") -> None:
        """Set minimal torch version according to Python actual version.

        >>> AssistantCLI.set_min_torch_by_python("../requirements/base.txt")

        """
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if py_ver not in LUT_PYTHON_TORCH:
            return
        with open(fpath) as fp:
            reqs = parse_requirements(fp.readlines())
        pkg_ver = next(p for p in reqs if p.name == "torch")
        pt_ver = min([parse(v[1]) for v in pkg_ver.specs])
        pt_ver = max(parse(LUT_PYTHON_TORCH[py_ver]), pt_ver)
        with open(fpath) as fp:
            requires = fp.read()
        requires = re.sub(r"torch>=[\d\.]+", f"torch>={pt_ver}", requires)
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(requires)

    @staticmethod
    def replace_min_requirements(fpath: str) -> None:
        """Replace all `>=` by `==` in given file."""
        logging.info(f"processing: {fpath}")
        with open(fpath, encoding="utf-8") as fp:
            req = fp.read()
        req = req.replace(">=", "==")
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(req)

    @staticmethod
    def set_oldest_versions(req_files: List[str] = REQUIREMENTS_FILES) -> None:
        """Set the oldest version for requirements."""
        AssistantCLI.set_min_torch_by_python()
        for fpath in req_files:
            logging.info(f"processing req: `{fpath}`")
            AssistantCLI.replace_min_requirements(fpath)

    @staticmethod
    def changed_domains(
        pr: Optional[int] = None,
        auth_token: Optional[str] = None,
        as_list: bool = False,
        general_sub_pkgs: Tuple[str] = _PKG_WIDE_SUBPACKAGES,
    ) -> Union[str, List[str]]:
        """Determine what domains were changed in particular PR."""
        if not pr:
            return "unittests"
        url = f"https://api.github.com/repos/Lightning-AI/torchmetrics/pulls/{pr}/files"
        logging.debug(url)
        data = request_url(url, auth_token)
        if not data:
            logging.debug("WARNING: No data was received -> test everything.")
            return "unittests"
        files = [d["filename"] for d in data]

        # filter out all integrations as they run in separate suit
        files = [fn for fn in files if not fn.startswith("tests/integrations")]
        if not files:
            logging.debug("Only integrations was changed so not reason for deep testing...")
            return ""
        # filter only docs files
        files_ = [fn for fn in files if fn.startswith("docs")]
        if len(files) == len(files_):
            logging.debug("Only docs was changed so not reason for deep testing...")
            return ""

        # filter only package files and skip inits
        _is_in_test = lambda fn: fn.startswith("tests")
        _filter_pkg = lambda fn: _is_in_test(fn) or (fn.startswith("src/torchmetrics") and "__init__.py" not in fn)
        files_pkg = [fn for fn in files if _filter_pkg(fn)]
        if not files_pkg:
            return "unittests"

        # parse domains
        def _crop_path(fname: str, paths: List[str]) -> str:
            for p in paths:
                fname = fname.replace(p, "")
            return fname

        files_pkg = [_crop_path(fn, ["src/torchmetrics/", "tests/unittests/", "functional/"]) for fn in files_pkg]
        # filter domain names
        tm_modules = [fn.split("/")[0] for fn in files_pkg if "/" in fn]
        # filter general (used everywhere) sub-packages
        tm_modules = [md for md in tm_modules if md not in general_sub_pkgs]
        if len(files_pkg) > len(tm_modules):
            logging.debug("Some more files was changed -> rather test everything...")
            return "unittests"
        # keep only unique
        if as_list:
            return list(tm_modules)
        tm_modules = [f"unittests/{md}" for md in set(tm_modules)]
        not_exists = [p for p in tm_modules if os.path.exists(p)]
        if not_exists:
            raise ValueError(f"Missing following paths: {not_exists}")
        return " ".join(tm_modules)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(AssistantCLI)
