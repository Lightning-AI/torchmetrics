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
from torch import Tensor
import torch
from torchmetrics.metric import Metric
from torchmetrics.functional.segmentation.mean_iou import _mean_iou_update, _mean_iou_compute, _mean_iou_validate_args


class MeanIoU(Metric):
    """Computes Mean Intersection over Union (mIoU) for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        per_class: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        _mean_iou_validate_args(num_classes, include_background, per_class)
        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class

        num_classes = num_classes - 1 if not include_background else num_classes
        self.add_state("score", default=torch.zeros(num_classes if per_class else 1), dist_reduce_fx="mean")
        self.add_state("num_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with the new data."""
        intersection, union = _mean_iou_update(preds, target, self.num_classes, self.include_background)
        score = _mean_iou_compute(intersection, union, per_class=self.per_class)
        self.score += score.mean(0) if self.per_class else score.mean()
        self.num_batches += 1

    def compute(self) -> Tensor:
        """Update the state with the new data."""
        return self.score / self.num_batches

