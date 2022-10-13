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

from torch import Tensor

from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn


class IOU(Metric):
    r"""Computes Intersection Over Union (IOU) Forward accepts two input tensors.

    One for preds boxes (Nx4) and one for target boxes (Mx4).
    Args:
        iou_threshold:
            threshold value of intersection over union. IOUs < threshold do not count toward the metric.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the ``allgather`` operation on the metric state. When ``None``, DDP
            will be used to perform the ``allgather``.
    """
    is_differentiable = False
    x: Tensor

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
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
        self.iou_threshold = iou_threshold
        self.add_state("x", default=Tensor([]), dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `IOU` will save all IOU > threshold values in buffer."
            " For large datasets with many objects, this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        x = _iou_update(preds, target, self.iou_threshold)
        self.x.append(x)

    def compute(self) -> Tensor:
        """Computes IOU based on inputs passed in to ``update`` previously."""
        return _iou_compute(self.x)
