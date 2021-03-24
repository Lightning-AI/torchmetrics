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
from distutils.version import LooseVersion

import torch

_TORCH_LOWER_1_4 = LooseVersion(torch.__version__) < LooseVersion("1.4.0")
_TORCH_LOWER_1_5 = LooseVersion(torch.__version__) < LooseVersion("1.5.0")
_TORCH_LOWER_1_6 = LooseVersion(torch.__version__) < LooseVersion("1.6.0")
_TORCH_GREATER_EQUAL_1_6 = LooseVersion(torch.__version__) >= LooseVersion("1.6.0")
