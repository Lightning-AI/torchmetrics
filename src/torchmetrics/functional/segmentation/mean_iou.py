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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _mean_iou_reshape_args(
    preds: Tensor,
    targets: Tensor,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tuple[Tensor, Tensor]:
    """Reshape tensors to 3D if needed."""
    if input_format == "one-hot":
        return preds, targets

    if preds.dim() == 1:
        preds = preds.unsqueeze(0).unsqueeze(0)
    elif preds.dim() == 2:
        preds = preds.unsqueeze(0)

    if targets.dim() == 1:
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif targets.dim() == 2:
        targets = targets.unsqueeze(0)
    return preds, targets


def _mean_iou_validate_args(
    num_classes: Optional[int],
    include_background: bool,
    per_class: bool,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> None:
    """Validate the arguments of the metric."""
    if input_format == "index" and num_classes is None:
        raise ValueError("Argument `num_classes` must be provided when `input_format='index'`.")
    if num_classes is not None and num_classes <= 0:
        raise ValueError(
            f"Expected argument `num_classes` must be `None` or a positive integer, but got {num_classes}."
        )
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` must be a boolean, but got {per_class}.")
    if input_format not in ["one-hot", "index"]:
        raise ValueError(f"Expected argument `input_format` to be one of 'one-hot', 'index', but got {input_format}.")


def _mean_iou_update(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    include_background: bool = False,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> tuple[Tensor, Tensor]:
    """Update the intersection and union counts for the mean IoU computation."""
    preds, target = _mean_iou_reshape_args(preds, target, input_format)
    _check_same_shape(preds, target)

    if input_format == "index":
        if num_classes is None:
            raise ValueError("Argument `num_classes` must be provided when `input_format='index'`.")
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)
    elif input_format == "one-hot" and num_classes is None:
        try:
            num_classes = preds.shape[1]
        except IndexError as err:
            raise IndexError(f"Cannot determine `num_classes` from `preds` tensor: {preds}.") from err
        if num_classes == 0:
            raise ValueError(f"Expected argument `num_classes` to be a positive integer, but got {num_classes}.")

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
    zero_division: Union[float, Literal["warn", "nan"]],
) -> Tensor:
    """Compute the mean IoU metric."""
    return _safe_divide(intersection, union, zero_division=zero_division)


def mean_iou(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    include_background: bool = True,
    per_class: bool = False,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    """Calculates the mean Intersection over Union (mIoU) for semantic segmentation.

    Returns -1 if class is completely absent both from predictions and ground truth labels.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes (required when input_format="index", optional when input_format="one-hot")
        include_background: Whether to include the background class in the computation
        per_class: Whether to compute the IoU for each class separately, else average over all classes
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
            or ``"index"`` for index tensors

    Returns:
        The mean IoU score

    Example:
        >>> import torch
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import mean_iou
        >>> # 4 samples, 5 classes, 16x16 prediction
        >>> preds = randint(0, 2, (4, 5, 16, 16), generator=torch.Generator().manual_seed(42))
        >>> # 4 samples, 5 classes, 16x16 target
        >>> target = randint(0, 2, (4, 5, 16, 16), generator=torch.Generator().manual_seed(43))
        >>> mean_iou(preds, target)
        tensor([0.3323, 0.3336, 0.3397, 0.3435])
        >>> mean_iou(preds, target, include_background=False, num_classes=5)
        tensor([0.3250, 0.3258, 0.3307, 0.3398])
        >>> mean_iou(preds, target, include_background=True, num_classes=5, per_class=True)
        tensor([[0.3617, 0.3128, 0.3366, 0.3242, 0.3263],
                [0.3646, 0.2893, 0.3297, 0.3073, 0.3770],
                [0.3756, 0.3168, 0.3505, 0.3400, 0.3155],
                [0.3579, 0.3317, 0.3797, 0.3523, 0.2957]])
        >>> # re-initialize tensors for ``input_format="index"``
        >>> preds = randint(0, 2, (4, 16, 16), generator=torch.Generator().manual_seed(42))
        >>> target = randint(0, 2, (4, 16, 16), generator=torch.Generator().manual_seed(43))
        >>> mean_iou(preds, target, num_classes=5, input_format = "index")
        tensor([0.3617, 0.3128, 0.3047, 0.3499])
        >>> mean_iou(preds, target, num_classes=5, per_class=True, input_format="index")
        tensor([[ 0.3617,  0.3617, -1.0000, -1.0000, -1.0000],
                [ 0.3128,  0.3128, -1.0000, -1.0000, -1.0000],
                [ 0.2727,  0.3366, -1.0000, -1.0000, -1.0000],
                [ 0.3756,  0.3242, -1.0000, -1.0000, -1.0000]])

    """
    _mean_iou_validate_args(num_classes, include_background, per_class, input_format)
    intersection, union = _mean_iou_update(preds, target, num_classes, include_background, input_format)
    scores = _mean_iou_compute(intersection, union, zero_division="nan")
    valid_classes = union > 0
    return scores.nan_to_num(-1.0) if per_class else scores.nansum(dim=-1) / valid_classes.sum(dim=-1)
