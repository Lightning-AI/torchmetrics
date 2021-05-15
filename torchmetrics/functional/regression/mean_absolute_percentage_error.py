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
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, tensor
from torch._C import dtype

from torchmetrics.utilities.checks import _check_same_shape


def _mean_absolute_percentage_error_update(preds: torch.Tensor, target: torch.Tensor, eps: torch.Tensor) -> Tuple[Tensor, int]:

    _check_same_shape(preds, target)


    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.max(eps, torch.abs(target))

    sum_abs_per_error = torch.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: int) -> Tensor:

    return sum_abs_per_error / num_obs


def mean_absolute_percentage_error(preds: torch.Tensor, target: torch.Tensor, eps: float= 1.17e-07) -> Tensor:
    """something"""

    eps = torch.tensor(eps)
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target, eps)
    mean_ape = _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)

    return mean_ape
