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
import logging
import os
from pathlib import Path
from typing import Optional, Union

import fire

_REQUEST_TIMEOUT = 10
_PATH_REPO_ROOT = Path(__file__).resolve().parent.parent
_PATH_DIR_TESTS = _PATH_REPO_ROOT / "tests"
_PKG_WIDE_SUBPACKAGES = ("utilities", "helpers")
_path_root = lambda *ds: os.path.join(_PATH_REPO_ROOT, *ds)
REQUIREMENTS_FILES = (*glob.glob(_path_root("requirements", "*.txt")), _path_root("requirements.txt"))


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
    def _replace_requirement(fpath: str, old_str: str = "", new_str: str = "") -> None:
        """Replace all strings given file."""
        logging.info(f"processing '{old_str}' -> '{new_str}': {fpath}")
        with open(fpath, encoding="utf-8") as fp:
            req = fp.read()
        req = req.replace(old_str, new_str)
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(req)

    @staticmethod
    def replace_str_requirements(old_str: str, new_str: str, req_files: list[str] = REQUIREMENTS_FILES) -> None:
        """Replace a particular string in all requirements files."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for fpath in req_files:
            AssistantCLI._replace_requirement(fpath, old_str=old_str, new_str=new_str)

    @staticmethod
    def replace_min_requirements(fpath: str) -> None:
        """Replace all `>=` by `==` in given file."""
        AssistantCLI._replace_requirement(fpath, old_str=">=", new_str="==")

    @staticmethod
    def set_oldest_versions(req_files: list[str] = REQUIREMENTS_FILES) -> None:
        """Set the oldest version for requirements."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for fpath in req_files:
            AssistantCLI.replace_min_requirements(fpath)

    @staticmethod
    def changed_domains(
        pr: Optional[int] = None,
        auth_token: Optional[str] = None,
        as_list: bool = False,
        general_sub_pkgs: tuple[str] = _PKG_WIDE_SUBPACKAGES,
    ) -> Union[str, list[str]]:
        """Determine what domains were changed in particular PR."""
        import github

        # define some edge case return cases
        _return_all = "unittests" if not as_list else ["torchmetrics"]
        _return_empty = [] if as_list else ""

        # early return if no PR number
        if not pr:
            return _return_all
        gh = github.Github(login_or_token=auth_token)
        pr = gh.get_repo("Lightning-AI/torchmetrics").get_pull(pr)
        files = [f.filename for f in pr.get_files()]

        # filter out all integrations as they run in separate suit
        files = [fn for fn in files if not fn.startswith("tests/integrations")]
        if not files:
            logging.debug("Only integrations was changed so not reason for deep testing...")
            return _return_empty

        # filter only docs files
        files_docs = [fn for fn in files if fn.startswith("docs")]
        if len(files) == len(files_docs):
            logging.debug("Only docs was changed so not reason for deep testing...")
            return _return_empty

        files_markdown = [fn for fn in files if fn.endswith(".md")]
        if len(files) == len(files_markdown):
            logging.debug("Only markdown files was changed so not reason for deep testing...")
            return _return_empty
        # filter out markdown files
        files = [fn for fn in files if fn not in files_markdown]

        # filter only testing files which are not specific tests so for example configurations or helper tools
        files_testing = [fn for fn in files if fn.startswith("tests") and not fn.endswith(".md") and "test_" not in fn]
        if files_testing:
            logging.debug("Some testing files was changed -> rather test everything...")
            return _return_all

        # files in requirements folder
        files_req = [fn for fn in files if fn.startswith("requirements")]
        req_domains = [fn.split("/")[1] for fn in files_req]
        # cleaning up determining domains
        req_domains = [req.replace(".txt", "").replace("_test", "") for req in req_domains if not req.endswith("_")]
        # if you touch base, you need to run everything
        if "base" in req_domains:
            return _return_all

        # filter only package files and skip inits
        _is_in_test = lambda fname: fname.startswith("tests")
        _filter_pkg = lambda fname: (
            _is_in_test(fname) or (fname.startswith("src/torchmetrics") and "__init__.py" not in fname)
        )
        files_pkg = [fn for fn in files if _filter_pkg(fn)]
        if not files_pkg:
            return _return_all

        # parse domains
        def _crop_path(fname: str, paths: tuple[str] = ("src/torchmetrics/", "tests/unittests/", "functional/")) -> str:
            for p in paths:
                fname = fname.replace(p, "")
            return fname

        files_pkg = [_crop_path(fn) for fn in files_pkg]
        # filter domain names
        tm_modules = [fn.split("/")[0] for fn in files_pkg if "/" in fn]
        # filter general (used everywhere) sub-packages
        tm_modules = [md for md in tm_modules if md not in general_sub_pkgs]
        if len(files_pkg) > len(tm_modules):
            logging.debug("Some more files was changed -> rather test everything...")
            return _return_all

        # compose the final list with requirements and touched modules
        test_modules = set(tm_modules + list(req_domains))
        if as_list:  # keep only unique
            return list(test_modules)

        test_modules = [os.path.join("unittests", fp) for fp in set(test_modules)]
        # filter only existing modules
        not_exists = [fp for fp in test_modules if not (_PATH_DIR_TESTS / fp).exists()]
        if not_exists:
            logging.debug(f"Missing following paths: {not_exists}")
        # filter only existing path in repo
        test_modules = [fp for fp in test_modules if (_PATH_DIR_TESTS / fp).exists()]
        if not test_modules:
            logging.debug("No tests were changed -> rather test everything...")
            return _return_all

        return " ".join(test_modules)

    @staticmethod
    def move_new_packages(dir_cache: str, dir_local: str, dir_staging: str) -> None:
        """Move unique packages from local folder to staging."""
        assert os.path.isdir(dir_cache), f"Missing folder with saved packages: '{dir_cache}'"  # noqa: S101
        assert os.path.isdir(dir_local), f"Missing folder with local packages: '{dir_local}'"  # noqa: S101
        assert os.path.isdir(dir_staging), f"Missing folder for staging: '{dir_staging}'"  # noqa: S101

        import shutil

        for pkg in os.listdir(dir_local):
            if not os.path.isfile(pkg):
                continue
            if pkg in os.listdir(dir_cache):
                continue
            logging.info(f"Moving '{pkg}' to staging...")
            shutil.move(os.path.join(dir_cache, pkg), os.path.join(dir_staging, pkg))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(AssistantCLI)
