# Copyright The Lightning team.
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
import random

import numpy
import torch
from lightning_utilities.core.imports import RequirementCache

from unittests._helpers.wrappers import skip_on_connection_issues, skip_on_cuda_oom, skip_on_running_out_of_memory


def seed_all(seed):
    """Set the seed of all computational frameworks."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


__all__ = ["seed_all", "skip_on_connection_issues", "skip_on_cuda_oom", "skip_on_running_out_of_memory"]


_SKLEARN_GREATER_EQUAL_1_3 = RequirementCache("scikit-learn>=1.3.0")
_SKLEARN_GREATER_EQUAL_1_7 = RequirementCache("scikit-learn>=1.7.0")
_TORCH_LESS_THAN_2_1 = RequirementCache("torch<2.1.0")
_TRANSFORMERS_RANGE_LT_4_50_LE_4_52 = RequirementCache("transformers>=4.50.0,<4.53.0")
