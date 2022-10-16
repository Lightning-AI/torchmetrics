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

from torchmetrics.detection.helpers import _fix_empty_tensors
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn


class BoxIntersectionOverUnion(Metric):
    r"""Computes Intersection Over Union (IoU). The forward method accepts two input tensors and the compute method
    returns a single tensor.

    One for preds boxes (Nx4) and one for target boxes (Mx4), expected format for both is `xyxy`.
    Args:
        iou_threshold:
            Optional threshold value of intersection over union. IOUs < threshold do not count toward the metric.
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    update_fn: Callable[[Tensor, Tensor, bool], Tensor] = _iou_update
    compute_fn: Callable[[Tensor], Tensor] = _iou_compute
    type: str = "iou"

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.add_state("x", default=Tensor([]), dist_reduce_fx="cat")

        rank_zero_warn(
            f"Metric `{self.type.upper()}` will save all {self.type.upper()} > threshold values in buffer."
            " For large datasets with many objects, this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predicted bounding boxes
            targets: Ground truth bounding boxes
        """
        if not isinstance(preds, Tensor):
            raise ValueError("Expected argument `preds` to be of type Tensor")
        if not isinstance(target, Tensor):
            raise ValueError("Expected argument `target` to be of type Tensor")
        if len(preds) != len(target):
            raise ValueError("Expected argument `preds` and `target` to have the same length")

        preds = _fix_empty_tensors(preds)
        target = _fix_empty_tensors(target)
        result = self.update_fn(preds, target, self.iou_threshold)
        self.x.append(result)

    def compute(self) -> Tensor:
        """Computes full IOU based on inputs passed in to ``update`` previously."""
        return self.x
