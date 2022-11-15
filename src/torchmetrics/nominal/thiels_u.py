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
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.nominal.thiels_u import _thiels_u_input_validation, theils_u
from torchmetrics.metric import Metric


class ThielsU(Metric):
    """Compute `ThielsU` statistic measuring the association between two categorical (nominal) data series.

    .. math::
        U(X|Y) = \frac{H(X) - H(X|Y)}{H(X)}

    where H(X) is entropy of variable X while H(X|Y) is the conditional entropy of X given Y

    Thiels's U is an asymmetric coefficient, i.e.

    .. math::
        V(preds, target) != V(target, preds)

    The output values lies in [0, 1]. 0 means y has no information about x while value 1 means y has complete
    information about x.

    Article: https://en.wikipedia.org/wiki/Uncertainty_coefficient

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``
        _PRECISION: used to round off values above 1 + _PRECISION and below 0 - _PRECISION
    Returns:
        Thiel's U Statistic: Tensor

    Example:
        >>> from torchmetrics import ThielsU
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> ThielsU()(preds, target)
        tensor(0.0853)
    """

    def __init__(
        self,
        reduction: Literal["sum", "mean"] = "mean",
        nan_strategy: Literal["replace", "drop"] = "replace",
        nan_replace_value: Optional[Union[int, float]] = 0.0,
        _PRECISION: float = 1e-13,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        _thiels_u_input_validation(nan_strategy, nan_replace_value)
        self.nan_strategy = nan_strategy
        self.nan_replace_value = nan_replace_value
        self._PRECISION = _PRECISION

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        self.add_state("statistic_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        """
        self.statistic_sum += theils_u(preds, target, self.nan_strategy, self.nan_replace_value)
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Computer Thiel's U statistic."""
        if self.reduction == "mean":
            return self.statistic_sum / self.total
        if self.reduction == "sum":
            return self.statistic_sum
