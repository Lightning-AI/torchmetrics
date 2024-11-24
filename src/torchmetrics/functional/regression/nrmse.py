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
from typing_extensions import Literal

from torchmetrics.functional.regression.mse import _mean_squared_error_update


def _normalized_root_mean_squared_error_update(
    preds: Tensor, target: Tensor, num_outputs: int, normalization: Literal["mean", "range", "std", "l2"] = "mean"
) -> tuple[Tensor, int, Tensor]:
    """Updates and returns the sum of squared errors and the number of observations for NRMSE computation.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting
        normalization: type of normalization to be applied. Choose from "mean", "range", "std", "l2"

    """
    sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs)

    target = target.view(-1) if num_outputs == 1 else target
    if normalization == "mean":
        denom = torch.mean(target, dim=0)
    elif normalization == "range":
        denom = torch.max(target, dim=0).values - torch.min(target, dim=0).values
    elif normalization == "std":
        denom = torch.std(target, correction=0, dim=0)
    elif normalization == "l2":
        denom = torch.norm(target, p=2, dim=0)
    else:
        raise ValueError(
            f"Argument `normalization` should be either 'mean', 'range', 'std' or 'l2' but got {normalization}"
        )
    return sum_squared_error, num_obs, denom


def _normalized_root_mean_squared_error_compute(
    sum_squared_error: Tensor, num_obs: Union[int, Tensor], denom: Tensor
) -> Tensor:
    """Calculates RMSE and normalizes it."""
    rmse = torch.sqrt(sum_squared_error / num_obs)
    return rmse / denom


def normalized_root_mean_squared_error(
    preds: Tensor,
    target: Tensor,
    normalization: Literal["mean", "range", "std", "l2"] = "mean",
    num_outputs: int = 1,
) -> Tensor:
    """Calculates the `Normalized Root Mean Squared Error`_ (NRMSE) also know as scatter index.

    Args:
        preds: estimated labels
        target: ground truth labels
        normalization: type of normalization to be applied. Choose from "mean", "range", "std", "l2" which corresponds
          to normalizing the RMSE by the mean of the target, the range of the target, the standard deviation of the
          target or the L2 norm of the target.
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with the NRMSE score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.regression import normalized_root_mean_squared_error
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> normalized_root_mean_squared_error(preds, target, normalization="mean")
        tensor(0.4000)
        >>> normalized_root_mean_squared_error(preds, target, normalization="range")
        tensor(0.2500)
        >>> normalized_root_mean_squared_error(preds, target, normalization="std")
        tensor(0.6030)
        >>> normalized_root_mean_squared_error(preds, target, normalization="l2")
        tensor(0.1667)

    Example (multioutput):
        >>> import torch
        >>> from torchmetrics.functional.regression import normalized_root_mean_squared_error
        >>> preds = torch.tensor([[0., 1], [2, 3], [4, 5], [6, 7]])
        >>> target = torch.tensor([[0., 1], [3, 3], [4, 5], [8, 9]])
        >>> normalized_root_mean_squared_error(preds, target, normalization="mean", num_outputs=2)
        tensor([0.2981, 0.2222])

    """
    sum_squared_error, num_obs, denom = _normalized_root_mean_squared_error_update(
        preds, target, num_outputs=num_outputs, normalization=normalization
    )
    return _normalized_root_mean_squared_error_compute(sum_squared_error, num_obs, denom)
