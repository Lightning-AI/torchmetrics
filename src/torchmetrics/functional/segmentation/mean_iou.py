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
from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _mean_iou_validate_args(
    num_classes: int,
    include_background: bool,
    per_class: bool,
) -> None:
    """Validate the arguments of the metric."""
    if num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` must be a boolean, but got {per_class}.")


def _mean_iou_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Update the intersection and union counts for the mean IoU computation."""
    _check_same_shape(preds, target)

    if (preds.bool() != preds).any():  # preds is an index tensor
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
    if (target.bool() != target).any():  # target is an index tensor
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if not include_background:
        preds, target = _ignore_background(preds, target)

    reduce_axis = list(range(2, preds.ndim))
    intersection = torch.sum(preds & target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    union = target_sum + pred_sum - intersection
    return intersection, union


def _mean_iou_compute(
    intersection: Tensor,
    union: Tensor,
    per_class: bool = False,
) -> Tensor:
    """Compute the mean IoU metric."""
    val = _safe_divide(intersection, union)
    return val if per_class else torch.mean(val, 1)


def mean_iou(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    per_class: bool = False,
) -> Tensor:
    """Calculates the mean Intersection over Union (mIoU) for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        per_class: Whether to compute the IoU for each class separately, else average over all classes

    Returns:
        The mean IoU score

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.functional.segmentation import mean_iou
        >>> preds = torch.randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = torch.randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> mean_iou(preds, target, num_classes=5)
        tensor([0.3193, 0.3305, 0.3382, 0.3246])
        >>> mean_iou(preds, target, num_classes=5, per_class=True)
        tensor([[0.3093, 0.3500, 0.3081, 0.3389, 0.2903],
                [0.2963, 0.3316, 0.3505, 0.2804, 0.3936],
                [0.3724, 0.3249, 0.3660, 0.3184, 0.3093],
                [0.3085, 0.3267, 0.3155, 0.3575, 0.3147]])

    """
    _mean_iou_validate_args(num_classes, include_background, per_class)
    intersection, union = _mean_iou_update(preds, target, num_classes, include_background)
    return _mean_iou_compute(intersection, union, per_class=per_class)
