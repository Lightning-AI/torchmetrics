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
from typing import Optional

import torch

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

if _TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_iou as tv_box_iou


def box_iou(
    preds: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Computes intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
    Example:
        >>> from torchmetrics.functional.detection import box_iou
        >>> preds = torch.Tensor([[100, 100, 200, 200]])
        >>> target = torch.Tensor([[110, 110, 210, 210]])
        >>> iou(preds, target)
        tensor([[0.6807]])
    Concerns:
    1) Currently, we do not check that the box label for each prediction matches the ground truth label of the
       overlapping box. If the labels do not match, then IoU should be 0, even if IoU > threshold
    2) We should not have a pred box match two different ground truth boxes. With N pred boxes and M
       ground truth boxes, the IoUs calc results in a NxM matrix and each row should have, AT MOST, 1 value > threshold.
       Do we want to check for this case?
    3) The result is a matrix NxM of IoU values. But this is not great as a metric. Should we sum this value, or
       take the mean? Perhaps we want to add a reduction function as input to result in a single value which a model
       can optimize for.
    4) This calculation cannot be run on a batch of data -> it is run independently for each image in the batch. Is
       this an issue?
    5) We will have a name collision with iou function in functional/classification/iou.py
    """
    iou = _box_iou_update(preds, target, iou_threshold)
    return _box_iou_compute(iou)


def _box_iou_update(preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float]) -> torch.Tensor:
    iou = tv_box_iou(preds, target)
    if iou_threshold is not None:
        return iou[iou >= iou_threshold]
    return iou


def _box_iou_compute(iou: torch.Tensor) -> torch.Tensor:
    return iou.sum()
