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
    cov_xy: Tensor,
    weights_prior: Tensor,
    num_outputs: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute weighted Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Updates are based on `Algorithms for calculating variance`_. Specifically, `online weighted variance`_ and
    `online weighted covariance`_.

    Variance intentionally not divided by sum of weights in `update` step as it is computed as necessary in
    the `compute` step.

    Args:
        preds: estimated scores
        target: ground truth scores
        weights: weight associated with scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        cov_xy: current covariance estimate between x and y tensor
        weights_prior: current sum of weights
        num_outputs: number of outputs in multioutput setting

    """
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    _check_data_shape_to_weights(preds, weights)

    if preds.ndim == 2:
        weights = weights.unsqueeze(1)  # singleton dimension for broadcasting

    weights_sum = weights.sum()

    if weights_prior > 0:  # True if prior observations exist
        mx_new = (weights_prior * mean_x + (weights * preds).sum(0)) / (weights_prior + weights_sum)
        my_new = (weights_prior * mean_y + (weights * target).sum(0)) / (weights_prior + weights_sum)

        var_x += (weights * (preds - mx_new) * (preds - mean_x)).sum(0)
        var_y += (weights * (target - my_new) * (target - mean_y)).sum(0)
    else:
        mx_new = ((weights * preds).sum(0) / weights_sum).to(mean_x.dtype)
        my_new = ((weights * target).sum(0) / weights_sum).to(mean_y.dtype)

        var_x = (weights * (preds - mx_new) ** 2).sum(0)
        var_y = (weights * (target - my_new) ** 2).sum(0)

    cov_xy += (weights * (preds - mx_new) * (target - my_new)).sum(0)

    return mx_new, my_new, var_x, var_y, cov_xy, weights_prior + weights_sum


def _weighted_pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    cov_xy: Tensor,
    weights_sum: Tensor,
) -> Tensor:
    """Compute the final weighted Pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        cov_xy: covariance estimate between x and y tensor
        weights_sum: sum of weights

    """
    # prevent overwrite the inputs
    var_x = var_x / weights_sum
    var_y = var_y / weights_sum
    cov_xy = cov_xy / weights_sum

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
    corrcoef = torch.full_like(cov_xy, float("nan"), device=cov_xy.device, dtype=cov_xy.dtype)
    valid_mask = ~zero_var_mask

    if valid_mask.any():
        corrcoef[valid_mask] = (
            (cov_xy[valid_mask] / (var_x[valid_mask] * var_y[valid_mask]).sqrt()).squeeze().to(corrcoef.dtype)
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
        >>> from torchmetrics.functional.regression import weighted_pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> weights = torch.tensor([0.2, 0.3, 0.5])
        >>> weighted_pearson_corrcoef(preds, target, weights)
        tensor(0.9849)

    Example (multi output weighted regression):
        >>> from torchmetrics.functional.regression import weighted_pearson_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> weights = torch.tensor([0.4, 0.6])
        >>> weighted_pearson_corrcoef(preds, target, weights)
        tensor([1., 1.])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, weights_sum = _temp.clone(), _temp.clone(), _temp.clone().sum()
    _, _, var_x, var_y, corr_xy, weights_sum = _weighted_pearson_corrcoef_update(
        preds,
        target,
        weights,
        mean_x,
        mean_y,
        var_x,
        var_y,
        corr_xy,
        weights_sum,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
    )
    return _weighted_pearson_corrcoef_compute(var_x, var_y, corr_xy, weights_sum)
