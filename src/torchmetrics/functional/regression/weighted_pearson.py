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
import math

import torch
from torch import Tensor

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs, _check_data_shape_to_weights
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape


def _weighted_pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    weights: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    num_prior: Tensor,
    num_outputs: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute weighted Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Updates are based on `Algorithms for calculating variance`_. Specifically, `online weighted variance`_ and
    `online weighted covariance`_.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        num_prior: current number of observed observations
        num_outputs: number of outputs in multioutput setting
        weights: weight associated with scores

    """
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    _check_data_shape_to_weights(preds, weights)

    num_obs = weights.sum()
    cond = num_prior.mean() > 0 or num_obs == 1  # True if prior observations exist

    if cond:
        mx_new = (num_prior * mean_x + preds.sum(0)) / (num_prior + num_obs)
        my_new = (num_prior * mean_y + target.sum(0)) / (num_prior + num_obs)
    else:
        mx_new = preds.mean(0).to(mean_x.dtype)
        my_new = target.mean(0).to(mean_y.dtype)

    num_prior += num_obs

    if cond:
        var_x += ((preds - mx_new) * (preds - mean_x)).sum(0)
        var_y += ((target - my_new) * (target - mean_y)).sum(0)
    else:
        var_x += preds.var(0) * (num_obs - 1)
        var_y += target.var(0) * (num_obs - 1)

    corr_xy += ((preds - mx_new) * (target - mean_y)).sum(0)
    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, num_prior


def _weighted_pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    nb: Tensor,
) -> Tensor:
    """Compute the final weighted Pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations

    """
    # prevent overwrite the inputs
    var_x = var_x / (nb - 1)
    var_y = var_y / (nb - 1)
    corr_xy = corr_xy / (nb - 1)
    # if var_x, var_y is float16 and on cpu, make it bfloat16 as sqrt is not supported for float16
    # on cpu, remove this after https://github.com/pytorch/pytorch/issues/54774 is fixed
    if var_x.dtype == torch.float16 and var_x.device == torch.device("cpu"):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()

    bound = math.sqrt(torch.finfo(var_x.dtype).eps)
    if (var_x < bound).any() or (var_y < bound).any():
        rank_zero_warn(
            "The variance of predictions or target is close to zero. This can cause instability in Pearson correlation"
            "coefficient, leading to wrong results. Consider re-scaling the input if possible or computing using a"
            f"larger dtype (currently using {var_x.dtype}). Setting the correlation coefficient to nan.",
            UserWarning,
        )

    zero_var_mask = (var_x < bound) | (var_y < bound)
    corrcoef = torch.full_like(corr_xy, float("nan"), device=corr_xy.device, dtype=corr_xy.dtype)
    valid_mask = ~zero_var_mask

    if valid_mask.any():
        corrcoef[valid_mask] = (
            (corr_xy[valid_mask] / (var_x[valid_mask] * var_y[valid_mask]).sqrt()).squeeze().to(corrcoef.dtype)
        )
        corrcoef = torch.clamp(corrcoef, -1.0, 1.0)
    return corrcoef.squeeze()


def weighted_pearson_corrcoef(preds: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Compute weighted Pearson correlation coefficient.

    Args:
        preds: torch.Tensor of shape (n_samples,) or (n_samples, n_outputs)
            Estimate scores
        target: torch.Tensor of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth scores
        weights: torch.Tensor of shape (n_samples,)
            Sample weights

    Example (single output weighted regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> weights = torch.tensor([0.1, 0.2, 0.5])
        >>> pearson(preds, target, weights)
        tensor(0.9849)

    Example (multi output weighted regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> weights = torch.tensor([0.1, 0.2])
        >>> pearson(preds, target, weights)
        tensor([1., 1.])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, var_x, var_y, corr_xy, nb = _weighted_pearson_corrcoef_update(
        preds,
        target,
        weights,
        mean_x,
        mean_y,
        var_x,
        var_y,
        corr_xy,
        nb,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
    )
    return _weighted_pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
