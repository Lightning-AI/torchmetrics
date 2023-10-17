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

from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8

if not _TORCHVISION_GREATER_EQUAL_0_8:
    __doctest_skip__ = ["generalized_intersection_over_union"]


def _giou_update(
    preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float], replacement_val: float = 0
) -> torch.Tensor:
    from torchvision.ops import generalized_box_iou

    iou = generalized_box_iou(preds, target)
    if iou_threshold is not None:
        iou[iou < iou_threshold] = replacement_val
    return iou


def _giou_compute(iou: torch.Tensor, aggregate: bool = True) -> torch.Tensor:
    if not aggregate:
        return iou
    return iou.diag().mean() if iou.numel() > 0 else torch.tensor(0.0, device=iou.device)


def generalized_intersection_over_union(
    preds: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: Optional[float] = None,
    replacement_val: float = 0,
    aggregate: bool = True,
) -> torch.Tensor:
    r"""Compute Generalized Intersection over Union (`GIOU`_) between two sets of boxes.

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
            Return the average value instead of the full matrix of values

    Example::
        By default giou is aggregated across all box pairs e.g. mean along the diagonal of the gIoU matrix:

        >>> import torch
        >>> from torchmetrics.functional.detection import generalized_intersection_over_union
        >>> preds = torch.tensor(
        ...     [
        ...         [296.55, 93.96, 314.97, 152.79],
        ...         [328.94, 97.05, 342.49, 122.98],
        ...         [356.62, 95.47, 372.33, 147.55],
        ...     ]
        ... )
        >>> target = torch.tensor(
        ...     [
        ...         [300.00, 100.00, 315.00, 150.00],
        ...         [330.00, 100.00, 350.00, 125.00],
        ...         [350.00, 100.00, 375.00, 150.00],
        ...     ]
        ... )
        >>> generalized_intersection_over_union(preds, target)
        tensor(0.5638)

    Example::
        By setting `aggregate=False` the full IoU matrix is returned:

        >>> import torch
        >>> from torchmetrics.functional.detection import generalized_intersection_over_union
        >>> preds = torch.tensor(
        ...     [
        ...         [296.55, 93.96, 314.97, 152.79],
        ...         [328.94, 97.05, 342.49, 122.98],
        ...         [356.62, 95.47, 372.33, 147.55],
        ...     ]
        ... )
        >>> target = torch.tensor(
        ...     [
        ...         [300.00, 100.00, 315.00, 150.00],
        ...         [330.00, 100.00, 350.00, 125.00],
        ...         [350.00, 100.00, 375.00, 150.00],
        ...     ]
        ... )
        >>> generalized_intersection_over_union(preds, target, aggregate=False)
        tensor([[ 0.6895, -0.4964, -0.4944],
                [-0.5105,  0.4673, -0.3434],
                [-0.6024, -0.4021,  0.5345]])

    """
    if not _TORCHVISION_GREATER_EQUAL_0_8:
        raise ModuleNotFoundError(
            f"`{generalized_intersection_over_union.__name__}` requires that `torchvision` version 0.8.0 or newer"
            " is installed."
            " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
        )
    iou = _giou_update(preds, target, iou_threshold, replacement_val)
    return _giou_compute(iou, aggregate)
