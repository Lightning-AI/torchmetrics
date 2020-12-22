#!/usr/bin/env python

import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

PATH_ROOT = os.path.dirname(__file__)

import pl_sandbox  # noqa: E402


def load_requirements(path_dir=PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [ln[:ln.index(comment_char)] if comment_char in ln else ln for ln in lines]
    reqs = [ln for ln in reqs if ln]
    return reqs


def load_long_describtion():
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(pl_sandbox.__homepage__, 'raw', pl_sandbox.__version__, 'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    return text


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='lightning-sandbox',
    version=pl_sandbox.__version__,
    description=pl_sandbox.__docs__,
    author=pl_sandbox.__author__,
    author_email=pl_sandbox.__author_email__,
    url=pl_sandbox.__homepage__,
    download_url='https://github.com/PyTorchLightning/lightning-sandbox',
    license=pl_sandbox.__license__,
    packages=find_packages(exclude=['tests', 'docs']),

    long_description=load_long_describtion(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=load_requirements(PATH_ROOT),

    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/lightning-sandbox/issues",
        "Documentation": "https://lightning-sandbox.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/lightning-sandbox",
    },

    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        # 'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
