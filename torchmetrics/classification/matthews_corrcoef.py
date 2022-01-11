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
from typing import Any, Callable, Optional

import torch
from deprecate import deprecated, void
from torch import Tensor

from torchmetrics.functional.classification.matthews_corrcoef import (
    _matthews_corrcoef_compute,
    _matthews_corrcoef_update,
)
from torchmetrics.metric import Metric


class MatthewsCorrCoef(Metric):
    r"""
    Calculates `Matthews correlation coefficient`_ that measures
    the general correlation or quality of a classification. In the binary case it
    is defined as:

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
        threshold:
            Threshold value for binary or multi-label probabilites.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Example:
        >>> from torchmetrics import MatthewsCorrCoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> matthews_corrcoef = MatthewsCorrCoef(num_classes=2)
        >>> matthews_corrcoef(preds, target)
        tensor(0.5774)

    """
    is_differentiable = False
    higher_is_better = True
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
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


class MatthewsCorrcoef(MatthewsCorrCoef):
    """Calculates `Matthews correlation coefficient`_ that measures the general correlation or quality of a
    classification.

    Example:
        >>> matthews_corrcoef = MatthewsCorrcoef(num_classes=2)
        >>> matthews_corrcoef(torch.tensor([0, 1, 0, 0]), torch.tensor([1, 1, 0, 0]))
        tensor(0.5774)

    .. deprecated:: v0.7
        Renamed in favor of :class:`torchmetrics.MatthewsCorrCoef`. Will be removed in v0.8.
    """

    @deprecated(target=MatthewsCorrCoef, deprecated_in="0.7", remove_in="0.8")
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        void(num_classes, threshold, compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
