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
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _generalized_dice_validate_args(
    num_classes: int,
    include_background: bool,
    per_class: bool,
    weight_type: Literal["square", "simple", "linear"],
    input_format: Literal["one-hot", "index"],
) -> None:
    """Validate the arguments of the metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` must be a boolean, but got {per_class}.")
    if weight_type not in ["square", "simple", "linear"]:
        raise ValueError(
            f"Expected argument `weight_type` to be one of 'square', 'simple', 'linear', but got {weight_type}."
        )
    if input_format not in ["one-hot", "index"]:
        raise ValueError(f"Expected argument `input_format` to be one of 'one-hot', 'index', but got {input_format}.")


def _generalized_dice_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    weight_type: Literal["square", "simple", "linear"] = "square",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tuple[Tensor, Tensor]:
    """Update the state with the current prediction and target."""
    _check_same_shape(preds, target)

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if preds.ndim < 3:
        raise ValueError(f"Expected both `preds` and `target` to have at least 3 dimensions, but got {preds.ndim}.")

    if not include_background:
        preds, target = _ignore_background(preds, target)

    reduce_axis = list(range(2, target.ndim))
    intersection = torch.sum(preds * target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    cardinality = target_sum + pred_sum
    if weight_type == "simple":
        weights = 1.0 / target_sum
    elif weight_type == "linear":
        weights = torch.ones_like(target_sum)
    elif weight_type == "square":
        weights = 1.0 / (target_sum**2)
    else:
        raise ValueError(
            f"Expected argument `weight_type` to be one of 'simple', 'linear', 'square', but got {weight_type}."
        )

    w_shape = weights.shape
    weights_flatten = weights.flatten()
    infs = torch.isinf(weights_flatten)
    weights_flatten[infs] = 0
    w_max = torch.max(weights, 0).values.repeat(w_shape[0], 1).T.flatten()
    weights_flatten[infs] = w_max[infs]
    weights = weights_flatten.reshape(w_shape)

    numerator = 2.0 * intersection * weights
    denominator = cardinality * weights
    return numerator, denominator


def _generalized_dice_compute(numerator: Tensor, denominator: Tensor, per_class: bool = True) -> Tensor:
    """Compute the generalized dice score."""
    if not per_class:
        numerator = torch.sum(numerator, 1)
        denominator = torch.sum(denominator, 1)
    return _safe_divide(numerator, denominator)


def generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    per_class: bool = False,
    weight_type: Literal["square", "simple", "linear"] = "square",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    """Compute the Generalized Dice Score for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        per_class: Whether to compute the score for each class separately, else average over all classes
        weight_type: Type of weight factor to apply to the classes. One of ``"square"``, ``"simple"``, or ``"linear"``
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
            or ``"index"`` for index tensors

    Returns:
        The Generalized Dice Score

    Example (with one-hot encoded tensors):
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import generalized_dice_score
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> generalized_dice_score(preds, target, num_classes=5)
        tensor([0.4830, 0.4935, 0.5044, 0.4880])
        >>> generalized_dice_score(preds, target, num_classes=5, per_class=True)
        tensor([[0.4724, 0.5185, 0.4710, 0.5062, 0.4500],
                [0.4571, 0.4980, 0.5191, 0.4380, 0.5649],
                [0.5428, 0.4904, 0.5358, 0.4830, 0.4724],
                [0.4715, 0.4925, 0.4797, 0.5267, 0.4788]])

    Example (with index tensors):
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import generalized_dice_score
        >>> preds = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> generalized_dice_score(preds, target, num_classes=5, input_format="index")
        tensor([0.1991, 0.1971, 0.2350, 0.2216])
        >>> generalized_dice_score(preds, target, num_classes=5, per_class=True, input_format="index")
        tensor([[0.1714, 0.2500, 0.1304, 0.2524, 0.2069],
                [0.1837, 0.2162, 0.0962, 0.2692, 0.1895],
                [0.3866, 0.1348, 0.2526, 0.2301, 0.2083],
                [0.1978, 0.2804, 0.1714, 0.1915, 0.2783]])

    """
    _generalized_dice_validate_args(num_classes, include_background, per_class, weight_type, input_format)
    numerator, denominator = _generalized_dice_update(
        preds, target, num_classes, include_background, weight_type, input_format
    )
    return _generalized_dice_compute(numerator, denominator, per_class)
