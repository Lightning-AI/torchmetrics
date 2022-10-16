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
from typing import Callable

from torch import Tensor

from torchmetrics.detection.box_iou import BoxIntersectionOverUnion
from torchmetrics.functional.detection.ciou import _ciou_compute, _ciou_update


class BoxCompleteIntersectionOverUnion(BoxIntersectionOverUnion):
    r"""Computes Complete Intersection Over Union (CIoU). The forward method accepts two input tensors and the
    compute method returns a single tensor.

    One for preds boxes (Nx4) and one for target boxes (Mx4), expected format for both is `xyxy`.
    Args:
        iou_threshold:
            Optional threshold value of intersection over union. IOUs < threshold do not count toward the metric.
    """

    update_fn: Callable[[Tensor, Tensor, bool], Tensor] = _ciou_update
    compute_fn: Callable[[Tensor], Tensor] = _ciou_compute
    type: str = "ciou"
