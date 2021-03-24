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

from pkg_resources import DistributionNotFound


def _compare_version(package: str, op, version) -> bool:
    """
    Compare package version with some requirements

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = LooseVersion(pkg.__version__)
    except AttributeError:
        return False
    if not (hasattr(pkg_version, "vstring") and hasattr(pkg_version, "version")):
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, LooseVersion(version))


_TORCH_LOWER_1_4 = _compare_version("torch", operator.lt, "1.4.0")
_TORCH_LOWER_1_5 = _compare_version("torch", operator.lt, "1.5.0")
_TORCH_LOWER_1_6 = _compare_version("torch", operator.lt, "1.6.0")
_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
