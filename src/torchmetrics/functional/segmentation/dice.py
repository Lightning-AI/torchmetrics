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
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _dice_score_validate_args(
    num_classes: int,
    include_background: bool,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> None:
    """Validate the arguments of the metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    allowed_average = ["micro", "macro", "weighted", "none"]
    if average is not None and average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} or None, but got {average}.")
    if input_format not in ["one-hot", "index"]:
        raise ValueError(f"Expected argument `input_format` to be one of 'one-hot', 'index', but got {input_format}.")


def _dice_score_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    _check_same_shape(preds, target)
    if preds.ndim < 3:
        raise ValueError(f"Expected both `preds` and `target` to have at least 3 dimensions, but got {preds.ndim}.")

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if not include_background:
        preds, target = _ignore_background(preds, target, num_classes)

    reduce_axis = list(range(2, preds.ndim))
    intersection = torch.sum(preds * target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)

    numerator = 2 * intersection
    denominator = pred_sum + target_sum
    return numerator, denominator


def _dice_score_compute(
    numerator: Tensor,
    denominator: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
) -> Tensor:
    if average == "micro":
        numerator = torch.sum(numerator, dim=1)
        denominator = torch.sum(denominator, dim=1)
    dice = _safe_divide(numerator, denominator, zero_division=1.0)
    if average == "macro":
        dice = torch.mean(dice)
    elif average == "weighted":
        weights = _safe_divide(denominator, torch.sum(denominator), zero_division=1.0)
        dice = torch.sum(dice * weights)
    return dice


def dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    """Compute the Dice score for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
            or ``"index"`` for index tensors

    Returns:
        The Dice score.

    Example (with one-hot encoded tensors):
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import dice_score
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> dice_score(preds, target, num_classes=5)
        tensor([0.4872, 0.5000, 0.5019, 0.4891, 0.4926])

    Example (with index tensors):

    """
    _dice_score_validate_args(num_classes, include_background, average, input_format)
    numerator, denominator = _dice_score_update(preds, target, num_classes, include_background, input_format)
    return _dice_score_compute(numerator, denominator, average)
