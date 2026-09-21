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
from typing import Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _segmentation_inputs_format, mask_edges, surface_distance
from torchmetrics.utilities.compute import _safe_divide


def _surface_dice_score_validate_args(
    num_classes: int,
    class_thresholds: Union[float, list[float], Tensor],
    include_background: bool,
    per_class: bool,
    spacing: Optional[Union[Tensor, list[float]]] = None,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> None:
    """Validate the arguments of the surface Dice score metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` to be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` to be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` to be a boolean, but got {per_class}.")
    if input_format not in ["one-hot", "index", "mixed"]:
        raise ValueError(
            f"Expected argument `input_format` to be one of 'one-hot', 'index', 'mixed', but got {input_format}."
        )
    if spacing is not None and not isinstance(spacing, (list, tuple, Tensor)):
        raise ValueError(f"Expected argument `spacing` to be a list, tuple or tensor, but got {type(spacing)}.")

    effective_classes = num_classes if include_background else num_classes - 1
    if effective_classes <= 0:
        raise ValueError("Expected at least one foreground class when `include_background=False`.")

    if isinstance(class_thresholds, (float, int)):
        if not torch.isfinite(torch.tensor(float(class_thresholds))) or class_thresholds < 0:
            raise ValueError("Expected scalar `class_thresholds` to be finite and non-negative.")
        return
    if isinstance(class_thresholds, Tensor):
        if class_thresholds.ndim > 1:
            raise ValueError("Expected tensor `class_thresholds` to be a scalar or 1D tensor.")
        if class_thresholds.numel() not in (1, effective_classes):
            raise ValueError(
                "Expected tensor `class_thresholds` to contain one threshold per evaluated class, but got "
                f"{class_thresholds.numel()} thresholds for {effective_classes} classes."
            )
        if not torch.isfinite(class_thresholds).all() or (class_thresholds < 0).any():
            raise ValueError("Expected all tensor `class_thresholds` to be finite and non-negative.")
        return
    if len(class_thresholds) != effective_classes:
        raise ValueError(
            "Expected `class_thresholds` to contain one threshold per evaluated class, but got "
            f"{len(class_thresholds)} thresholds for {effective_classes} classes."
        )
    threshold_tensor = torch.tensor(class_thresholds, dtype=torch.float32)
    if not torch.isfinite(threshold_tensor).all() or (threshold_tensor < 0).any():
        raise ValueError("Expected all `class_thresholds` to be finite and non-negative.")


def _surface_dice_score_prepare_class_thresholds(
    class_thresholds: Union[float, list[float], Tensor],
    num_classes: int,
    include_background: bool,
    device: torch.device,
) -> Tensor:
    """Broadcast class thresholds to the evaluated class dimension."""
    effective_classes = num_classes if include_background else num_classes - 1
    if isinstance(class_thresholds, Tensor):
        thresholds = class_thresholds.to(device=device, dtype=torch.float32).flatten()
        if thresholds.numel() == 1:
            thresholds = thresholds.repeat(effective_classes)
    elif isinstance(class_thresholds, (float, int)):
        thresholds = torch.full((effective_classes,), float(class_thresholds), device=device, dtype=torch.float32)
    else:
        thresholds = torch.tensor(class_thresholds, device=device, dtype=torch.float32)
    return thresholds


def _surface_dice_score_prepare_spacing(
    spacing: Optional[Union[Tensor, list[float]]],
    spatial_dims: int,
) -> Union[tuple[float, float], tuple[float, float, float]]:
    """Convert spacing to a tuple matching the number of spatial dimensions."""
    if spacing is None:
        if spatial_dims == 2:
            return (1.0, 1.0)
        return (1.0, 1.0, 1.0)
    if isinstance(spacing, Tensor):
        spacing = spacing.detach().cpu().flatten().tolist()
    if len(spacing) != spatial_dims:
        raise ValueError(f"Expected argument `spacing` to have length {spatial_dims} but got {len(spacing)}.")
    if spatial_dims == 2:
        return float(spacing[0]), float(spacing[1])
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def _surface_dice_score_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    class_thresholds: Union[float, list[float], Tensor],
    include_background: bool,
    spacing: Optional[Union[Tensor, list[float]]] = None,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> tuple[Tensor, Tensor]:
    """Calculate per-sample and per-class surface Dice scores and a validity mask."""
    preds, target = _segmentation_inputs_format(preds, target, include_background, num_classes, input_format)

    if preds.ndim not in (4, 5):
        raise ValueError(
            "Expected `preds` and `target` to have 2D or 3D spatial dimensions after formatting, "
            f"but got tensors with shape {preds.shape}."
        )

    preds = preds.bool()
    target = target.bool()
    thresholds = _surface_dice_score_prepare_class_thresholds(
        class_thresholds, num_classes, include_background, preds.device
    )
    spacing_tuple = _surface_dice_score_prepare_spacing(spacing, preds.ndim - 2)
    spacing_list = list(spacing_tuple)

    score = torch.full((preds.shape[0], preds.shape[1]), float("nan"), device=preds.device)
    valid = torch.zeros((preds.shape[0], preds.shape[1]), device=preds.device, dtype=torch.bool)

    for b in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            pred_mask = preds[b, c]
            target_mask = target[b, c]
            if not pred_mask.any() and not target_mask.any():
                continue

            edges_pred, edges_target, areas_pred, areas_target = mask_edges(  # type: ignore[misc]
                pred_mask, target_mask, spacing=spacing_tuple
            )
            edges_pred = edges_pred.bool()
            edges_target = edges_target.bool()
            surfel_areas_pred = areas_pred[edges_pred]
            surfel_areas_target = areas_target[edges_target]
            total_area = surfel_areas_pred.sum() + surfel_areas_target.sum()
            if total_area <= 0:
                continue

            dist_pred_to_target = surface_distance(edges_pred, edges_target, spacing=spacing_list)
            dist_target_to_pred = surface_distance(edges_target, edges_pred, spacing=spacing_list)
            overlap_pred = surfel_areas_pred[dist_pred_to_target <= thresholds[c]].sum()
            overlap_target = surfel_areas_target[dist_target_to_pred <= thresholds[c]].sum()
            score[b, c] = (overlap_pred + overlap_target) / total_area
            valid[b, c] = True

    return score, valid


def _surface_dice_score_compute(score: Tensor, valid: Tensor, per_class: bool) -> Tensor:
    """Reduce surface Dice scores across classes when requested."""
    if per_class:
        return score
    return _safe_divide(torch.nan_to_num(score, nan=0.0).sum(dim=-1), valid.sum(dim=-1), zero_division="nan")


def surface_dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    class_thresholds: Union[float, list[float], Tensor],
    include_background: bool = False,
    per_class: bool = False,
    spacing: Optional[Union[Tensor, list[float]]] = None,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> Tensor:
    r"""Compute the normalized Surface Dice score for semantic segmentation.

    Surface Dice measures how much predicted and target boundaries overlap within a class-specific tolerance. It is
    useful when contour alignment matters more than region overlap and is often reported as normalized surface Dice
    (NSD) in medical segmentation benchmarks.

    A class score is computed by summing the surface length or area that lies within the acceptable tolerance on both
    segmentations and dividing by the total surface length or area:

    .. math::
        \operatorname{NSD}(Y, \hat{Y}) =
        \frac{|D_Y'| + |D_{\hat{Y}}'|}{|D_Y| + |D_{\hat{Y}}|}

    where :math:`D_Y` and :math:`D_{\hat{Y}}` are the boundary elements of the target and prediction, and
    :math:`D_Y'` and :math:`D_{\hat{Y}}'` are the subsets whose closest boundary distance is below the corresponding
    class threshold.

    This implementation supports 2D and 3D segmentation masks. Classes that are absent in both prediction and target
    return ``nan`` in the per-class output and are ignored when reducing across classes. The thresholds are expressed
    in pixels or voxels when ``spacing`` is not provided, and in physical units when ``spacing`` is provided.
    For 3D inputs, the closest-surface distance computation currently relies on SciPy-backed distance transforms.

    Args:
        preds: Predictions from model.
        target: Ground truth values.
        num_classes: Number of classes in the segmentation problem.
        class_thresholds: Either a single non-negative tolerance shared by all evaluated classes or one tolerance per
            evaluated class. When ``include_background=False``, the number of thresholds must match
            ``num_classes - 1``.
        include_background: Whether to include the background class in the computation.
        per_class: Whether to return class-wise scores instead of reducing across classes.
        spacing: Pixel or voxel spacing along each spatial dimension. If not provided, unit spacing is assumed.
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor.

    Returns:
        If ``per_class=False``, a tensor of shape ``(N,)`` with one score per sample. If ``per_class=True``, a tensor
        of shape ``(N, C)`` with one score per sample and evaluated class.

    Example (with one-hot encoded tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import surface_dice_score
        >>> preds = torch.zeros(2, 2, 8, 8, dtype=torch.int)
        >>> target = torch.zeros(2, 2, 8, 8, dtype=torch.int)
        >>> preds[:, 1, 2:6, 2:6] = 1
        >>> target[:, 1, 2:6, 2:6] = 1
        >>> surface_dice_score(preds, target, num_classes=2, class_thresholds=1.0)
        tensor([1., 1.])

    Example (with index tensors):
        >>> import torch
        >>> from torchmetrics.functional.segmentation import surface_dice_score
        >>> preds = torch.zeros(2, 8, 8, dtype=torch.long)
        >>> target = torch.zeros(2, 8, 8, dtype=torch.long)
        >>> preds[:, 2:6, 2:6] = 1
        >>> target[:, 2:6, 2:6] = 1
        >>> surface_dice_score(preds, target, num_classes=2, class_thresholds=1.0, input_format="index")
        tensor([1., 1.])

    """
    _surface_dice_score_validate_args(
        num_classes=num_classes,
        class_thresholds=class_thresholds,
        include_background=include_background,
        per_class=per_class,
        spacing=spacing,
        input_format=input_format,
    )
    score, valid = _surface_dice_score_update(
        preds=preds,
        target=target,
        num_classes=num_classes,
        class_thresholds=class_thresholds,
        include_background=include_background,
        spacing=spacing,
        input_format=input_format,
    )
    return _surface_dice_score_compute(score, valid, per_class)
