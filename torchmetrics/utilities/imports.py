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
import operator
from importlib import import_module
from importlib.util import find_spec

from packaging.version import Version
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
    """
    Compare package version with some requirements

    >>> import operator
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_LOWER_1_4 = _compare_version("torch", operator.lt, "1.4.0")
_TORCH_LOWER_1_5 = _compare_version("torch", operator.lt, "1.5.0")
_TORCH_LOWER_1_6 = _compare_version("torch", operator.lt, "1.6.0")
_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
