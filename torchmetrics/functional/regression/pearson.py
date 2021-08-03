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


def _pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    n_prior: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute Pearson Correlation Coefficient. Checks for same shape of
    input tensors.

    Args:
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        n_prior: current number of observed observations
    """
    # Data checking
    _check_same_shape(preds, target)
    preds = preds.squeeze()
    target = target.squeeze()
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError("Expected both predictions and target to be 1 dimensional tensors.")

    n_obs = preds.numel()
    mx_new = (n_prior * mean_x + preds.mean() * n_obs) / (n_prior + n_obs)
    my_new = (n_prior * mean_y + target.mean() * n_obs) / (n_prior + n_obs)
    n_prior += n_obs
    var_x += ((preds - mx_new) * (preds - mean_x)).sum()
    var_y += ((target - my_new) * (target - mean_y)).sum()
    corr_xy += ((preds - mx_new) * (target - mean_y)).sum()
    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, n_prior


def _pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    nb: Tensor,
) -> Tensor:
    """Computes the final pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations
    """
    var_x /= nb - 1
    var_y /= nb - 1
    corr_xy /= nb - 1
    corrcoef = (corr_xy / (var_x * var_y).sqrt()).squeeze()
    return torch.clamp(corrcoef, -1.0, 1.0)


def pearson_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """Computes pearson correlation coefficient.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example:
        >>> from torchmetrics.functional import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target)
        tensor(0.9849)
    """
    _temp = torch.zeros(1, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, var_x, var_y, corr_xy, nb = _pearson_corrcoef_update(preds, target, mean_x, mean_y, var_x, var_y, corr_xy, nb)
    return _pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
