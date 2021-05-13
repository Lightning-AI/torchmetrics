import re
import sys

LUT_PYTHON_TORCH = {
    '3.8': '1.4',
    '3.9': '1.7.1',
}


def set_min_torch_by_python(fpath: str = 'requirements.txt') -> None:
    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}'
    if py_ver not in LUT_PYTHON_TORCH:
        return
    req = re.sub(r'torch>=[\d\.]+', f'torch>={LUT_PYTHON_TORCH[py_ver]}', open(fpath).read())
    open(fpath, 'w').write(req)


def replace_min_requirements(fpath: str) -> None:
    req = open(fpath).read().replace('>=', '==')
    open(fpath, 'w').write(req)


if __name__ == '__main__':
    set_min_torch_by_python()
    for fpath in ('requirements.txt', 'requirements/test.txt', 'requirements/integrate.txt'):
        replace_min_requirements(fpath)
