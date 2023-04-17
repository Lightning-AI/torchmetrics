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


def _mean_squared_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Squared Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    diff = preds - target
    sum_squared_error = torch.sum(diff * diff)
    n_obs = target.numel()
    return sum_squared_error, n_obs


def _mean_squared_error_compute(sum_squared_error: Tensor, n_obs: Union[int, Tensor], squared: bool = True) -> Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_squared_error, n_obs = _mean_squared_error_update(preds, target)
        >>> _mean_squared_error_compute(sum_squared_error, n_obs)
        tensor(0.2500)
    """
    return sum_squared_error / n_obs if squared else torch.sqrt(sum_squared_error / n_obs)


def mean_squared_error(preds: Tensor, target: Tensor, squared: bool = True) -> Tensor:
    """Compute mean squared error.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False

    Return:
        Tensor with MSE

    Example:
        >>> from torchmetrics.functional.regression import mean_squared_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_squared_error(x, y)
        tensor(0.2500)
    """
    sum_squared_error, n_obs = _mean_squared_error_update(preds, target)
    return _mean_squared_error_compute(sum_squared_error, n_obs, squared=squared)
