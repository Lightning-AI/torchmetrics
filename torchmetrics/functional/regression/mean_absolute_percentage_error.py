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

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-06,
) -> Tuple[Tensor, int]:

    _check_same_shape(preds, target)

    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

    sum_abs_per_error = torch.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: int) -> Tensor:
    return sum_abs_per_error / num_obs


def mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    """
    Computes mean absolute percentage error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAPE

    Note:
        The epsilon value is taken from `scikit-learn's
        implementation
        <https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_regression.py#L197>`_.

    Example:
        >>> from torchmetrics.functional import mean_absolute_percentage_error
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> mean_absolute_percentage_error(preds, target)
        tensor(0.2667)
    """
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
    mean_ape = _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)

    return mean_ape
