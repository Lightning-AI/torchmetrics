import glob
import logging
import os
import re
import sys

LUT_PYTHON_TORCH = {
    "3.8": "1.4",
    "3.9": "1.7.1",
}
REQUIREMENTS_FILES = ("requirements.txt",) + tuple(glob.glob(os.path.join("requirements", "*.txt")))


def set_min_torch_by_python(fpath: str = "requirements.txt") -> None:
    """set minimal torch version"""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py_ver not in LUT_PYTHON_TORCH:
        return
    with open(fpath) as fp:
        req = fp.read()
    req = re.sub(r"torch>=[\d\.]+", f"torch>={LUT_PYTHON_TORCH[py_ver]}", req)
    with open(fpath, "w") as fp:
        fp.write(req)


def replace_min_requirements(fpath: str) -> None:
    """replace all `>=` by `==` in given file"""
    logging.info(f"processing: {fpath}")
    with open(fpath) as fp:
        req = fp.read()
    req = req.replace(">=", "==")
    with open(fpath, "w") as fp:
        fp.write(req)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_min_torch_by_python()
    for fpath in REQUIREMENTS_FILES:
        replace_min_requirements(fpath)
