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

from torchmetrics.functional.nominal.cramers import _cramers_v_compute, _cramers_v_update
from torchmetrics.metric import Metric


class CramersV(Metric):
    r"""Compute `Cramer's V`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        V = \sqrt{\frac{\chi^2 / 2}{\min(r - 1, k - 1)}}

    where

    .. math::
        \chi^2 = \sum_{i,j} \ frac{\left(n_{ij} - \frac{n_{i.} n_{.j}}{n}\right)^2}{\frac{n_{i.} n_{.j}}{n}}

    Cramer's V is a symmetric coefficient, i.e.

    .. math::
        V(preds, target) = V(target, preds)

    The output values lies in [0, 1].

    Args:
        num_classes: Integer specifing the number of classes
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Cramer's V statistic

    Example:
        >>> from torchmetrics import CramersV
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(0, 4, (100,))
        >>> target = torch.round(preds + torch.randn(100)).clamp(0, 4)
        >>> cramers_v = CramersV(num_classes=5)
        >>> cramers_v(preds, target)
        tensor(0.5284))
    """

    is_differentiable = False
    higher_is_better = False
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        bias_correction: bool = True,
        nan_strategy: Literal["replace", "drop"] = "replace",
        nan_replace_value: Optional[Union[int, float]] = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bias_correction = bias_correction

        if nan_strategy not in ["replace", "drop"]:
            raise ValueError(
                f"Argument `nan_strategy` is expected to be one of `['replace', 'drop']`, but got {nan_strategy}"
            )
        if nan_strategy == "replace" and not isinstance(nan_replace_value, (int, float)):
            raise ValueError(
                "Argument `nan_replace` is expected to be of a type `int` or `float` when `nan_strategy = 'replace`, "
                f"but got {nan_replace_value}"
            )
        self.nan_strategy = nan_strategy
        self.nan_replace_value = nan_replace_value

        self.add_state("confmat", torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Tensor with predictions
            target: Tensor with true labels
        """
        confmat = _cramers_v_update(preds, target, self.num_classes, self.nan_strategy, self.nan_replace_value)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computer Cramer's V statistic."""
        return _cramers_v_compute(self.confmat, self.bias_correction)
