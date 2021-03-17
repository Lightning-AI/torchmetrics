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
"""Import utilities"""
import importlib
import operator
from distutils.version import LooseVersion
from importlib.util import find_spec

from pkg_resources import DistributionNotFound


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment
    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op, version) -> bool:
    """Compare package version with some requirements
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    if not _module_available(package):
        return False
    try:
        pkg = importlib.import_module(package)
        assert hasattr(pkg, '__version__')
        pkg_version = pkg.__version__
        return op(pkg_version, LooseVersion(version))
    except DistributionNotFound:
        return False


_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
