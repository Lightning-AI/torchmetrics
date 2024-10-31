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

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["intersection_over_union"]


def _iou_update(
    preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float], replacement_val: float = 0
) -> torch.Tensor:
    """Compute the IoU matrix between two sets of boxes."""
    if preds.ndim != 2 or preds.shape[-1] != 4:
        raise ValueError(f"Expected preds to be of shape (N, 4) but got {preds.shape}")
    if target.ndim != 2 or target.shape[-1] != 4:
        raise ValueError(f"Expected target to be of shape (N, 4) but got {target.shape}")

    from torchvision.ops import box_iou

    if preds.numel() == 0:  # if no boxes are predicted
        return torch.zeros(target.shape[0], target.shape[0], device=target.device, dtype=torch.float32)
    if target.numel() == 0:  # if no boxes are true
        return torch.zeros(preds.shape[0], preds.shape[0], device=preds.device, dtype=torch.float32)

    iou = box_iou(preds, target)
    if iou_threshold is not None:
        iou[iou < iou_threshold] = replacement_val
    return iou


def _iou_compute(iou: torch.Tensor, aggregate: bool = True) -> torch.Tensor:
    if not aggregate:
        return iou
    return iou.diag().mean() if iou.numel() > 0 else torch.tensor(0.0, device=iou.device)


def intersection_over_union(
    preds: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: Optional[float] = None,
    replacement_val: float = 0,
    aggregate: bool = True,
) -> torch.Tensor:
    r"""Compute Intersection over Union between two sets of boxes.

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
        By default iou is aggregated across all box pairs e.g. mean along the diagonal of the IoU matrix:

        >>> import torch
        >>> from torchmetrics.functional.detection import intersection_over_union
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
        >>> intersection_over_union(preds, target)
        tensor(0.5879)

    Example::
        By setting `aggregate=False` the full IoU matrix is returned:

        >>> import torch
        >>> from torchmetrics.functional.detection import intersection_over_union
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
        >>> intersection_over_union(preds, target, aggregate=False)
        tensor([[0.6898, 0.0000, 0.0000],
                [0.0000, 0.5086, 0.0000],
                [0.0000, 0.0000, 0.5654]])

    """
    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError(
            f"`{intersection_over_union.__name__}` requires that `torchvision` is installed."
            " Please install with `pip install torchmetrics[detection]`."
        )
    iou = _iou_update(preds, target, iou_threshold, replacement_val)
    return _iou_compute(iou, aggregate)
