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
    mx: Tensor,
    my: Tensor,
    vx: Tensor,
    vy: Tensor,
    cxy: Tensor,
    n: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ updates current estimates of the mean, cov and n_obs with new data for calculating pearsons correlation """
    # Data checking
    _check_same_shape(preds, target)
    preds = preds.squeeze()
    target = target.squeeze()
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError('Expected both predictions and target to be 1 dimensional tensors.')

    n_obs = preds.numel()
    mx_new = (n * mx + preds.mean() * n_obs) / (n + n_obs)
    my_new = (n * my + target.mean() * n_obs) / (n + n_obs)
    n += n_obs
    vx += ((preds - mx_new) * (preds - mx)).sum()
    vy += ((target - my_new) * (target - my)).sum()
    cxy += ((preds - mx_new) * (target - my)).sum()
    mx = mx_new
    my = my_new

    return mx, my, vx, vy, cxy, n


def _pearson_corrcoef_compute(
    vx: Tensor,
    vy: Tensor,
    cxy: Tensor,
    n: Tensor,
) -> Tensor:
    """ computes the final pearson correlation based on covariance matrix and number of observatiosn """
    vx /= (n - 1)
    vy /= (n - 1)
    cxy /= (n - 1)
    corrcoef = cxy / (vx * vy).sqrt()
    return torch.clamp(corrcoef, -1.0, 1.0)


def pearson_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """
    Computes pearson correlation coefficient.

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
    mx, my, vx = _temp.clone(), _temp.clone(), _temp.clone()
    vy, cxy, n = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, vx, vy, cxy, n = _pearson_corrcoef_update(preds, target, mx, my, vx, vy, cxy, n)
    return _pearson_corrcoef_compute(vx, vy, cxy, n)
