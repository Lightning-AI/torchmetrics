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
from distutils.version import LooseVersion

from pkg_resources import DistributionNotFound, get_distribution


def _compare_version(package: str, op, version) -> bool:
    try:
        pkg_version = LooseVersion(get_distribution(package).version)
        return op(pkg_version, LooseVersion(version))
    except DistributionNotFound:
        return False


_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
