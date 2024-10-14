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

from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.segmentation.utils import check_if_binarized, surface_distance
from torchmetrics.utilities.checks import _check_same_shape


def _hausdorff_distance_validate_args(
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
) -> None:
    """Validate the arguments of `hausdorff_distance` function."""
    if distance_metric not in ["euclidean", "chessboard", "taxicab"]:
        raise ValueError(
            f"Arg `distance_metric` must be one of 'euclidean', 'chessboard', 'taxicab', but got {distance_metric}."
        )
    if spacing is not None and not isinstance(spacing, (list, Tensor)):
        raise ValueError(f"Arg `spacing` must be a list or tensor, but got {type(spacing)}.")


def _hausdorff_distance_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute `Hausdorff Distance`_.

    Args:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map

    Returns:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map

    """
    check_if_binarized(preds)
    check_if_binarized(target)
    _check_same_shape(preds, target)
    return preds, target


def _hausdorff_distance_compute(
    preds: Tensor,
    target: Tensor,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
) -> Tensor:
    """Compute `Hausdorff Distance`_.

    Args:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map
        distance_metric: distance metric to calculate surface distance. One of `["euclidean", "chessboard", "taxicab"]`.
        spacing: spacing between pixels along each spatial dimension

    Returns:
        Hausdorff distance

    Example:
        >>> preds = torch.tensor([[1, 1, 1, 1, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 1, 1, 1, 1]], dtype=torch.bool)
        >>> target = torch.tensor([[1, 1, 1, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 1, 1, 1, 0]], dtype=torch.bool)
        >>> hausdorff_distance(preds, target, distance_metric="euclidean")
        tensor(1.)

    """
    fwd = surface_distance(preds, target, distance_metric=distance_metric, spacing=spacing)
    bwd = surface_distance(target, preds, distance_metric=distance_metric, spacing=spacing)
    return torch.max(torch.tensor([fwd.max(), bwd.max()]))


def hausdorff_distance(
    preds: Tensor,
    target: Tensor,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
) -> Tensor:
    """Calculate `Hausdorff Distance`_.

    Args:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map
        distance_metric: distance metric to calculate surface distance. One of `["euclidean", "chessboard", "taxicab"]`.
        spacing: spacing between pixels along each spatial dimension

    Returns:
        Hausdorff Distance

    Example:
        >>> import torch
        >>> from torchmetrics.functional.segmentation import hausdorff_distance
        >>> preds = torch.tensor([[1, 1, 1, 1, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 1, 1, 1, 1]], dtype=torch.bool)
        >>> target = torch.tensor([[1, 1, 1, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 1, 1, 1, 0]], dtype=torch.bool)
        >>> hausdorff_distance(preds, target, distance_metric="euclidean")
        tensor(1.)

    """
    _hausdorff_distance_validate_args(distance_metric, spacing)
    preds, target = _hausdorff_distance_update(preds, target)
    return _hausdorff_distance_compute(preds, target, distance_metric=distance_metric, spacing=spacing)
