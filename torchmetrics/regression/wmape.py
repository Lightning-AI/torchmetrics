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
from typing import Any

import torch
from torch import Tensor

from torchmetrics.functional.regression.wmape import (
    _weighted_mean_absolute_percentage_error_compute,
    _weighted_mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric


class WeightedMeanAbsolutePercentageError(Metric):
    r"""
    Computes weighted mean absolute percentage error (`WMAPE`_). The output of WMAPE metric
    is a non-negative floating point, where the optimal value is 0. It is computes as:

    .. math::
        \text{WMAPE} = \frac{\sum_{t=1}^n | y_t - \hat{y}_t | }{\sum_{t=1}^n |y_t| }

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randn(20,)
        >>> target = torch.randn(20,)
        >>> metric = WeightedMeanAbsolutePercentageError()
        >>> metric(preds, target)
        tensor(1.3967)

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    sum_scale: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_scale", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.sum_scale += sum_scale

    def compute(self) -> Tensor:
        """Computes weighted mean absolute percentage error over state."""
        return _weighted_mean_absolute_percentage_error_compute(self.sum_abs_error, self.sum_scale)
