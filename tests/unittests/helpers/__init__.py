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
import random

import numpy
import torch

from torchmetrics.utilities.imports import _TORCH_LOWER_1_4, _TORCH_LOWER_1_5, _TORCH_LOWER_1_6

_MARK_TORCH_MIN_1_4 = dict(condition=_TORCH_LOWER_1_4, reason="required PT >= 1.4")
_MARK_TORCH_MIN_1_5 = dict(condition=_TORCH_LOWER_1_5, reason="required PT >= 1.5")
_MARK_TORCH_MIN_1_6 = dict(condition=_TORCH_LOWER_1_6, reason="required PT >= 1.6")


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
