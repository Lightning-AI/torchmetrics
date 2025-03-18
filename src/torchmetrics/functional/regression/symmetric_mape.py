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
from typing import Union

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _symmetric_mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-06,
) -> tuple[Tensor, int]:
    """Update and returns variables required to compute Symmetric Mean Absolute Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Avoids ``ZeroDivisionError``.

    """
    _check_same_shape(preds, target)

    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target) + torch.abs(preds), min=epsilon)

    sum_abs_per_error = 2 * torch.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _symmetric_mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute Symmetric Mean Absolute Percentage Error.

    Args:
        sum_abs_per_error: Sum of values of symmetric absolute percentage errors over all observations
            ``(symmetric absolute percentage error = 2 * |target - prediction| / (target + prediction))``
        num_obs: Number of predictions or observations

    Example:
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(preds, target)
        >>> _symmetric_mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
        tensor(0.2290)

    """
    return sum_abs_per_error / num_obs


def symmetric_mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    r"""Compute symmetric mean absolute percentage error (SMAPE_).

    .. math:: \text{SMAPE} = \frac{2}{n}\sum_1^n\frac{|   y_i - \hat{y_i} |}{max(| y_i | + | \hat{y_i} |, \epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with SMAPE.

    Example:
        >>> from torchmetrics.functional.regression import symmetric_mean_absolute_percentage_error
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> symmetric_mean_absolute_percentage_error(preds, target)
        tensor(0.2290)

    """
    sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(
        preds,
        target,
    )
    return _symmetric_mean_absolute_percentage_error_compute(
        sum_abs_per_error,
        num_obs,
    )
