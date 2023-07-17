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
from typing import Tuple, Union

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _mean_absolute_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    sum_abs_error = torch.sum(torch.abs(preds - target))
    n_obs = target.numel()
    return sum_abs_error, n_obs


def _mean_absolute_error_compute(sum_abs_error: Tensor, n_obs: Union[int, Tensor]) -> Tensor:
    """Compute Mean Absolute Error.

    Args:
        sum_abs_error: Sum of absolute value of errors over all observations
        n_obs: Number of predictions or observations

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)
        >>> _mean_absolute_error_compute(sum_abs_error, n_obs)
        tensor(0.2500)
    """
    return sum_abs_error / n_obs


def mean_absolute_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute mean absolute error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAE

    Example:
        >>> from torchmetrics.functional.regression import mean_absolute_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_absolute_error(x, y)
        tensor(0.2500)
    """
    sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)
    return _mean_absolute_error_compute(sum_abs_error, n_obs)
