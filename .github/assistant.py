import glob
import logging
import os
import re
import sys
from typing import List

import fire

LUT_PYTHON_TORCH = {
    "3.8": "1.4",
    "3.9": "1.7.1",
}
_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
REQUIREMENTS_FILES = (os.path.join(_PATH_ROOT, "requirements.txt"),) + tuple(glob.glob(os.path.join(_PATH_ROOT, "requirements", "*.txt")))


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
        """Set minimal torch version accrding to Python actual version."""
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(AssistantCLI)
