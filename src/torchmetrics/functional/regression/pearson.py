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
from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape


def _pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    n_prior: Tensor,
    num_outputs: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        n_prior: current number of observed observations
        num_outputs: Number of outputs in multioutput setting
    """
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    cond = n_prior.mean() > 0

    n_obs = preds.shape[0]
    if cond:
        mx_new = (n_prior * mean_x + preds.sum(0)) / (n_prior + n_obs)
        my_new = (n_prior * mean_y + target.sum(0)) / (n_prior + n_obs)
    else:
        mx_new = preds.mean(0)
        my_new = target.mean(0)

    n_prior += n_obs

    if cond:
        var_x += ((preds - mx_new) * (preds - mean_x)).sum(0)
        var_y += ((target - my_new) * (target - mean_y)).sum(0)

    else:
        var_x += preds.var(0) * (n_obs - 1)
        var_y += target.var(0) * (n_obs - 1)
    corr_xy += ((preds - mx_new) * (target - mean_y)).sum(0)
    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, n_prior


def _pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    nb: Tensor,
) -> Tensor:
    """Compute the final pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations
    """
    var_x /= nb - 1
    var_y /= nb - 1
    corr_xy /= nb - 1
    # if var_x, var_y is float16 and on cpu, make it bfloat16 as sqrt is not supported for float16
    # on cpu, remove this after https://github.com/pytorch/pytorch/issues/54774 is fixed
    if var_x.dtype == torch.float16 and var_x.device == torch.device("cpu"):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()

    corrcoef = (corr_xy / (var_x * var_y).sqrt()).squeeze()
    return torch.clamp(corrcoef, -1.0, 1.0)


def pearson_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """Compute pearson correlation coefficient.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> pearson_corrcoef(preds, target)
        tensor([1., 1.])
    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, var_x, var_y, corr_xy, nb = _pearson_corrcoef_update(
        preds, target, mean_x, mean_y, var_x, var_y, corr_xy, nb, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
