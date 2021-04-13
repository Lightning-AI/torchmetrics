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
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    old_mean: Optional[Tensor] = None,
    old_cov: Optional[Tensor] = None,
    old_nobs: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ updates current estimates of the mean, cov and n_obs with new data for calculating
        pearsons correlation
    """
    # Data checking
    _check_same_shape(preds, target)
    preds = preds.squeeze()
    target = target.squeeze()
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError('Expected both predictions and target to be 1 dimensional tensors.')

    return preds, target


def _pearson_corrcoef_compute(preds: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """ computes the final pearson correlation based on covariance matrix and number of observatiosn """
    preds_diff = preds - preds.mean()
    target_diff = target - target.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    denom = preds_std * target_std
    # prevent division by zero
    if denom == 0:
        denom += eps

    corrcoef = cov / denom
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
    preds, target = _pearson_corrcoef_update(preds, target)
    return _pearson_corrcoef_compute(preds, target)
