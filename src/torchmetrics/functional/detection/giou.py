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
    from torchvision.ops import generalized_box_iou
else:
    generalized_box_iou = None
    __doctest_skip__ = ["generalized_intersection_over_union"]

__doctest_requires__ = {("generalized_intersection_over_union",): ["torchvision"]}


def _giou_update(
    preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float], replacement_val: float = 0
) -> torch.Tensor:
    iou = generalized_box_iou(preds, target)
    if iou_threshold is not None:
        iou[iou < iou_threshold] = replacement_val
    return iou


def _giou_compute(iou: torch.Tensor, labels_eq: bool = True) -> torch.Tensor:
    if labels_eq:
        return iou.diag().mean()
    return iou.mean()


def generalized_intersection_over_union(
    preds: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: Optional[float] = None,
    replacement_val: float = 0,
    aggregate: bool = True,
) -> torch.Tensor:
    r"""Compute `Generalized Intersection over Union <https://arxiv.org/abs/1902.09630>`_ between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.

    Args:
        preds:
            The input tensor containing the predicted bounding boxes.
        target:
            The tensor containing the ground truth.
        iou_threshold:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        replacement_val:
            Value to replace values under the threshold with.
        aggregate:
            Return the average value instead of the complete IoU matrix.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.detection import generalized_intersection_over_union
        >>> preds = torch.Tensor([[100, 100, 200, 200]])
        >>> target = torch.Tensor([[110, 110, 210, 210]])
        >>> generalized_intersection_over_union(preds, target)
        tensor(0.6641)
    """
    if not _TORCHVISION_GREATER_EQUAL_0_8:
        raise ModuleNotFoundError(
            f"`{generalized_intersection_over_union.__name__}` requires that `torchvision` version 0.8.0 or newer"
            " is installed."
            " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
        )
    iou = _giou_update(preds, target, iou_threshold, replacement_val)
    return _giou_compute(iou) if aggregate else iou
