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

from torchmetrics.functional.classification.matthews_corrcoef import (
    _matthews_corrcoef_compute,
    _matthews_corrcoef_update,
)
from torchmetrics.metric import Metric


class MatthewsCorrCoef(Metric):
    r"""Calculates `Matthews correlation coefficient`_ that measures the general correlation or quality of a classification.

    In the binary case it is defined as:

    .. math::
        MCC = \frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}

    where TP, TN, FP and FN are respectively the true postitives, true negatives,
    false positives and false negatives. Also works in the case of multi-label or
    multi-class input.

    Note:
        This metric produces a multi-dimensional output, so it can not be directly logged.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        threshold: Threshold value for binary or multi-label probabilites.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MatthewsCorrCoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> matthews_corrcoef = MatthewsCorrCoef(num_classes=2)
        >>> matthews_corrcoef(preds, target)
        tensor(0.5774)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _matthews_corrcoef_update(preds, target, self.num_classes, self.threshold)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes matthews correlation coefficient."""
        return _matthews_corrcoef_compute(self.confmat)
