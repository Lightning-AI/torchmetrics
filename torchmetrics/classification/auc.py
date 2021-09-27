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
from typing import Any, Callable, List, Optional

from torch import Tensor

from torchmetrics.functional.classification.auc import _auc_compute, _auc_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class AUC(Metric):
    r"""
    Computes Area Under the Curve (AUC) using the trapezoidal rule

    Forward accepts two input tensors that should be 1D and have the same number
    of elements

    Args:
        reorder: AUC expects its first input to be sorted. If this is not the case,
            setting this argument to ``True`` will use a stable sorting algorithm to
            sort the input in descending order
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the ``allgather`` operation on the metric state. When ``None``, DDP
            will be used to perform the ``allgather``.
    """
    x: List[Tensor]
    y: List[Tensor]

    def __init__(
        self,
        reorder: bool = False,
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

        self.reorder = reorder

        self.add_state("x", default=[], dist_reduce_fx="cat")
        self.add_state("y", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AUC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        x, y = _auc_update(preds, target)

        self.x.append(x)
        self.y.append(y)

    def compute(self) -> Tensor:
        """Computes AUC based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return _auc_compute(x, y, reorder=self.reorder)

    @property
    def is_differentiable(self) -> bool:
        """AUC metrics is considered as non differentiable so it should have `false` value for `is_differentiable`
        property."""
        return False
