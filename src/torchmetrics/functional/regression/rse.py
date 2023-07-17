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

from torchmetrics.functional.regression.r2 import _r2_score_update


def _relative_squared_error_compute(
    sum_squared_obs: Tensor,
    sum_obs: Tensor,
    sum_squared_error: Tensor,
    n_obs: Union[int, Tensor],
    squared: bool = True,
) -> Tensor:
    """Computes Relative Squared Error.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        sum_squared_error: Residual sum of squares
        n_obs: Number of predictions or observations
        squared: Returns RRSE value if set to False.

    Example:
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> # RSE uses the same update function as R2 score.
        >>> sum_squared_obs, sum_obs, rss, n_obs = _r2_score_update(preds, target)
        >>> _relative_squared_error_compute(sum_squared_obs, sum_obs, rss, n_obs, squared=True)
        tensor(0.0632)
    """
    epsilon = torch.finfo(sum_squared_error.dtype).eps
    rse = sum_squared_error / torch.clamp(sum_squared_obs - sum_obs * sum_obs / n_obs, min=epsilon)
    if not squared:
        rse = torch.sqrt(rse)
    return torch.mean(rse)


def relative_squared_error(preds: Tensor, target: Tensor, squared: bool = True) -> Tensor:
    r"""Computes the relative squared error (RSE).

    .. math:: \text{RSE} = \frac{\sum_i^N(y_i - \hat{y_i})^2}{\sum_i^N(y_i - \overline{y})^2}

    Where :math:`y` is a tensor of target values with mean :math:`\overline{y}`, and
    :math:`\hat{y}` is a tensor of predictions.

    If `preds` and `targets` are 2D tensors, the RSE is averaged over the second dim.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RRSE value if set to False
    Return:
        Tensor with RSE

    Example:
        >>> from torchmetrics.functional.regression import relative_squared_error
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> relative_squared_error(preds, target)
        tensor(0.0514)
    """
    sum_squared_obs, sum_obs, rss, n_obs = _r2_score_update(preds, target)
    return _relative_squared_error_compute(sum_squared_obs, sum_obs, rss, n_obs, squared=squared)
