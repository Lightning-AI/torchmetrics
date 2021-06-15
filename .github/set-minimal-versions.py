import os
import re
import sys

LUT_PYTHON_TORCH = {
    '3.8': '1.4',
    '3.9': '1.7.1',
}
REQUIREMENTS_FILES = (
    'requirements.txt',
    os.path.join('requirements', 'test.txt'),
    os.path.join('requirements', 'integrate.txt'),
)


def set_min_torch_by_python(fpath: str = 'requirements.txt') -> None:
    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}'
    if py_ver not in LUT_PYTHON_TORCH:
        return
    with open(fpath) as fp:
        req = fp.read()
    req = re.sub(r'torch>=[\d\.]+', f'torch>={LUT_PYTHON_TORCH[py_ver]}', req)
    with open(fpath, 'w') as fp:
        fp.write(req)


def replace_min_requirements(fpath: str) -> None:
    with open(fpath) as fp:
        req = fp.read()
    req = req.replace('>=', '==')
    with open(fpath, 'w') as fp:
        fp.write(req)


if __name__ == '__main__':
    set_min_torch_by_python()
    for fpath in REQUIREMENTS_FILES:
        replace_min_requirements(fpath)
