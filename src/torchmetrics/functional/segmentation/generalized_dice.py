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

from torchmetrics.functional.segmentation.utils import _segmentation_inputs_format
from torchmetrics.utilities.compute import _safe_divide


def _generalized_dice_validate_args(
    num_classes: int,
    include_background: bool,
    per_class: bool,
    weight_type: Literal["square", "simple", "linear"],
    input_format: Literal["one-hot", "index", "mixed"],
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
    if input_format not in ["one-hot", "index", "mixed"]:
        raise ValueError(
            f"Expected argument `input_format` to be one of 'one-hot', 'index', 'mixed', but got {input_format}."
        )


def _generalized_dice_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    weight_type: Literal["square", "simple", "linear"] = "square",
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update the state with the current prediction and target.

    Returns:
        numerator: per-sample per-class numerator
        denominator: per-sample per-class denominator
        support: per-sample per-class boolean indicating if class is present in target

    """
    preds, target = _segmentation_inputs_format(preds, target, include_background, num_classes, input_format)

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

    # Track which classes are present in the target (support)
    support = target_sum > 0

    return numerator, denominator, support


def _generalized_dice_compute(
    numerator: Tensor,
    denominator: Tensor,
    per_class: bool = True,
    support: Tensor | None = None,
) -> Tensor:
    """Compute the generalized dice score."""
    if not per_class:
        numerator = torch.sum(numerator, 1)
        denominator = torch.sum(denominator, 1)
        return _safe_divide(numerator, denominator)

    # For per_class=True, support is required
    if support is None:
        raise ValueError("support must be provided when per_class=True")

    # For per_class=True, compute score per sample per class
    score = _safe_divide(numerator, denominator, zero_division="nan")

    # Average over samples where class is present (support)
    # For samples without support, score is NaN, so we use nansum and divide by support count
    support_count = support.sum(dim=0).float()
    # Replace 0 support count with NaN to get NaN result for absent classes
    support_count[support_count == 0] = float("nan")
    return torch.nansum(score, dim=0) / support_count


def generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    per_class: bool = False,
    weight_type: Literal["square", "simple", "linear"] = "square",
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> Tensor:
    """Compute the Generalized Dice Score for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        per_class: Whether to compute the score for each class separately, else average over all classes
        weight_type: Type of weight factor to apply to the classes. One of ``"square"``, ``"simple"``, or ``"linear"``
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor

    Returns:
        The Generalized Dice Score

    Example (with one-hot encoded tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import generalized_dice_score
        >>> preds = torch.tensor([
        ...     [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]]],
        ...     [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]],
        ...     [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 0]]],
        ...     [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [1, 1]]],
        ... ])
        >>> target = torch.tensor([
        ...     [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]]],
        ...     [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]],
        ...     [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 0]]],
        ...     [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [1, 1]]],
        ... ])
        >>> generalized_dice_score(preds, target, num_classes=5)
        tensor([1., 1., 1., 1.])
        >>> generalized_dice_score(preds, target, num_classes=5, per_class=True)
        tensor([1., 1., 1., 1., 1.])

    Example (with index tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import generalized_dice_score
        >>> preds = torch.tensor([
        ...     [[0, 1], [2, 3]],
        ...     [[1, 2], [3, 4]],
        ...     [[2, 3], [4, 0]],
        ...     [[3, 4], [0, 1]],
        ... ])
        >>> target = torch.tensor([
        ...     [[0, 1], [2, 3]],
        ...     [[1, 2], [3, 4]],
        ...     [[2, 3], [4, 0]],
        ...     [[3, 4], [0, 1]],
        ... ])
        >>> generalized_dice_score(preds, target, num_classes=5, input_format="index")
        tensor([1., 1., 1., 1.])
        >>> generalized_dice_score(preds, target, num_classes=5, per_class=True, input_format="index")
        tensor([1., 1., 1., 1., 1.])

    """
    _generalized_dice_validate_args(num_classes, include_background, per_class, weight_type, input_format)
    numerator, denominator, support = _generalized_dice_update(
        preds, target, num_classes, include_background, weight_type, input_format
    )
    return _generalized_dice_compute(numerator, denominator, per_class=per_class, support=support)
