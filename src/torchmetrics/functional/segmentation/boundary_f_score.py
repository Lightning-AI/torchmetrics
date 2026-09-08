# Copyright The Lightning team.
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
import torch
from torch import Tensor
from torch.nn.functional import max_pool2d, max_pool3d
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _segmentation_inputs_format, binary_erosion
from torchmetrics.utilities.compute import _safe_divide


def _boundary_f_score_validate_args(
    num_classes: int,
    include_background: bool,
    per_class: bool,
    boundary_width: int,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> None:
    """Validate the arguments of the boundary F-score metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` to be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` to be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` to be a boolean, but got {per_class}.")
    if not isinstance(boundary_width, int) or boundary_width < 0:
        raise ValueError(f"Expected argument `boundary_width` to be a non-negative integer, but got {boundary_width}.")
    if input_format not in ["one-hot", "index", "mixed"]:
        raise ValueError(
            f"Expected argument `input_format` to be one of 'one-hot', 'index', 'mixed', but got {input_format}."
        )


def _boundary_f_score_extract_boundaries(mask: Tensor) -> Tensor:
    """Extract mask boundaries using binary erosion."""
    spatial_shape = mask.shape[2:]
    mask = mask.bool()
    eroded = binary_erosion(mask.reshape(-1, 1, *spatial_shape).byte()).reshape_as(mask).bool()
    return mask & ~eroded


def _boundary_f_score_dilate_boundaries(boundaries: Tensor, boundary_width: int) -> Tensor:
    """Dilate boundaries by a pixel tolerance."""
    if boundary_width == 0:
        return boundaries

    kernel_size = 2 * boundary_width + 1
    if boundaries.ndim == 4:
        return max_pool2d(boundaries.float(), kernel_size=kernel_size, stride=1, padding=boundary_width) > 0
    if boundaries.ndim == 5:
        return max_pool3d(boundaries.float(), kernel_size=kernel_size, stride=1, padding=boundary_width) > 0
    raise ValueError(
        "Expected `preds` and `target` to have 2D or 3D spatial dimensions after formatting, "
        f"but got tensors with shape {boundaries.shape}."
    )


def _boundary_f_score_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    boundary_width: int,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> tuple[Tensor, Tensor]:
    """Calculate per-sample and per-class boundary F-scores and a validity mask."""
    preds, target = _segmentation_inputs_format(preds, target, include_background, num_classes, input_format)

    if preds.ndim not in (4, 5):
        raise ValueError(
            "Expected `preds` and `target` to have 2D or 3D spatial dimensions after formatting, "
            f"but got tensors with shape {preds.shape}."
        )

    preds = preds.bool()
    target = target.bool()

    pred_boundaries = _boundary_f_score_extract_boundaries(preds)
    target_boundaries = _boundary_f_score_extract_boundaries(target)

    dilated_pred_boundaries = _boundary_f_score_dilate_boundaries(pred_boundaries, boundary_width)
    dilated_target_boundaries = _boundary_f_score_dilate_boundaries(target_boundaries, boundary_width)

    reduce_axis = tuple(range(2, preds.ndim))
    pred_boundary_area = pred_boundaries.sum(dim=reduce_axis)
    target_boundary_area = target_boundaries.sum(dim=reduce_axis)
    matched_pred_boundary = (pred_boundaries & dilated_target_boundaries).sum(dim=reduce_axis)
    matched_target_boundary = (target_boundaries & dilated_pred_boundaries).sum(dim=reduce_axis)

    precision = _safe_divide(matched_pred_boundary, pred_boundary_area, zero_division=0.0)
    recall = _safe_divide(matched_target_boundary, target_boundary_area, zero_division=0.0)
    score = _safe_divide(2 * precision * recall, precision + recall, zero_division=0.0)

    valid = (pred_boundary_area > 0) | (target_boundary_area > 0)
    score = score.masked_fill(~valid, float("nan"))
    return score, valid


def _boundary_f_score_compute(score: Tensor, valid: Tensor, per_class: bool) -> Tensor:
    """Reduce boundary F-scores across classes when requested."""
    if per_class:
        return score
    return _safe_divide(torch.nan_to_num(score, nan=0.0).sum(dim=-1), valid.sum(dim=-1), zero_division="nan")


def boundary_f_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    per_class: bool = False,
    boundary_width: int = 1,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> Tensor:
    """Compute the Boundary F-score for semantic segmentation.

    Boundary F-score evaluates how well predicted object contours align with target contours. A predicted boundary
    pixel counts as correct if a target boundary pixel exists within ``boundary_width`` pixels, and vice versa. The
    final score is the harmonic mean of boundary precision and boundary recall.

    This implementation supports 2D and 3D segmentation masks. Classes that are absent in both prediction and target
    return ``nan`` in the per-class output and are ignored when reducing across classes. The tolerance is expressed in
    pixels for 2D masks and voxels for 3D volumes.

    Args:
        preds: Predictions from model.
        target: Ground truth values.
        num_classes: Number of classes in the segmentation problem.
        include_background: Whether to include the background class in the computation.
        per_class: Whether to return class-wise scores instead of reducing across classes.
        boundary_width: Integer pixel tolerance used when matching predicted and target boundaries.
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor.

    Returns:
        If ``per_class=False``, a tensor of shape ``(N,)`` with one score per sample. If ``per_class=True``, a tensor
        of shape ``(N, C)`` with one score per sample and class.

    Example (with one-hot encoded tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import boundary_f_score
        >>> preds = torch.zeros(2, 3, 8, 8, dtype=torch.int)
        >>> target = torch.zeros(2, 3, 8, 8, dtype=torch.int)
        >>> preds[:, 1, 2:6, 2:6] = 1
        >>> target[:, 1, 2:6, 2:6] = 1
        >>> boundary_f_score(preds, target, num_classes=3)
        tensor([1., 1.])

    Example (with index tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import boundary_f_score
        >>> preds = torch.zeros(2, 8, 8, dtype=torch.long)
        >>> target = torch.zeros(2, 8, 8, dtype=torch.long)
        >>> preds[:, 2:6, 2:6] = 1
        >>> target[:, 2:6, 2:6] = 1
        >>> boundary_f_score(preds, target, num_classes=2, input_format="index")
        tensor([1., 1.])

    Example (tolerance sensitivity):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import boundary_f_score
        >>> preds = torch.zeros(1, 8, 8, dtype=torch.long)
        >>> target = torch.zeros(1, 8, 8, dtype=torch.long)
        >>> preds[:, 2:6, 2:6] = 1
        >>> target[:, 2:6, 3:7] = 1
        >>> strict = boundary_f_score(preds, target, num_classes=2, input_format="index", boundary_width=0)
        >>> tolerant = boundary_f_score(preds, target, num_classes=2, input_format="index", boundary_width=1)
        >>> bool(strict.item() < tolerant.item())
        True

    """
    _boundary_f_score_validate_args(num_classes, include_background, per_class, boundary_width, input_format)
    score, valid = _boundary_f_score_update(
        preds,
        target,
        num_classes=num_classes,
        include_background=include_background,
        boundary_width=boundary_width,
        input_format=input_format,
    )
    return _boundary_f_score_compute(score, valid, per_class)
