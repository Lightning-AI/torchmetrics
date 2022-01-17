# Copyright The PyTorch Lightning team.
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

_REQUEST_TIMEOUT = 10
_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
_PKG_WIDE_SUBPACKAGES = ("utilities",)
LUT_PYTHON_TORCH = {
    "3.8": "1.4",
    "3.9": "1.7.1",
}
REQUIREMENTS_FILES = (os.path.join(_PATH_ROOT, "requirements.txt"),) + tuple(
    glob.glob(os.path.join(_PATH_ROOT, "requirements", "*.txt"))
)


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
    @staticmethod
    def prune_packages(req_file: str, *pkgs: str) -> None:
        """Prune packages from requirement file."""
        with open(req_file) as fp:
            lines = fp.readlines()

        for pkg in pkgs:
            lines = [ln for ln in lines if not ln.startswith(pkg)]
        logging.info(lines)

        with open(req_file, "w") as fp:
            fp.writelines(lines)

    @staticmethod
    def set_min_torch_by_python(fpath: str = "requirements.txt") -> None:
        """Set minimal torch version according to Python actual version."""
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if py_ver not in LUT_PYTHON_TORCH:
            return
        with open(fpath) as fp:
            req = fp.read()
        req = re.sub(r"torch>=[\d\.]+", f"torch>={LUT_PYTHON_TORCH[py_ver]}", req)
        with open(fpath, "w") as fp:
            fp.write(req)

    @staticmethod
    def replace_min_requirements(fpath: str) -> None:
        """Replace all `>=` by `==` in given file."""
        logging.info(f"processing: {fpath}")
        with open(fpath) as fp:
            req = fp.read()
        req = req.replace(">=", "==")
        with open(fpath, "w") as fp:
            fp.write(req)

    @staticmethod
    def set_oldest_versions(req_files: List[str] = REQUIREMENTS_FILES) -> None:
        AssistantCLI.set_min_torch_by_python()
        for fpath in req_files:
            AssistantCLI.replace_min_requirements(fpath)

    @staticmethod
    def changed_domains(
        pr: int,
        auth_token: Optional[str] = None,
        as_list: bool = False,
        general_sub_pkgs: Tuple[str] = _PKG_WIDE_SUBPACKAGES,
    ) -> Union[str, List[str]]:
        """Determine what domains were changed in particular PR."""
        url = f"https://api.github.com/repos/PyTorchLightning/metrics/pulls/{pr}/files"
        logging.debug(url)
        data = request_url(url, auth_token)
        if not data:
            logging.error("No data was received.")
            return "tests"
        files = [d["filename"] for d in data]
        # filter only package files and skip inits
        files = [fn for fn in files if fn.startswith("torchmetrics") and "__init__.py" not in fn]
        if not files:
            return "tests"
        # parse domains
        files = [fn.replace("torchmetrics/", "").replace("functional/", "") for fn in files]
        # filter domain names
        tm_modules = [fn.split("/")[0] for fn in files if "/" in fn]
        # filter general (used everywhere) sub-packages
        tm_modules = [md for md in tm_modules if md not in general_sub_pkgs]
        if len(files) > len(tm_modules):
            return "tests"
        # keep only unique
        tm_modules = set(tm_modules)
        if as_list:
            return list(tm_modules)
        return " ".join([f"tests/{md}" for md in tm_modules])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(AssistantCLI)
