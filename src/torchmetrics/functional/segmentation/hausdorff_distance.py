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

from typing import Literal, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.segmentation.utils import (
    _ignore_background,
    edge_surface_distance,
)
from torchmetrics.utilities.checks import _check_same_shape


def _hausdorff_distance_validate_args(
    num_classes: int,
    include_background: bool,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
    directed: bool = False,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> None:
    """Validate the arguments of `hausdorff_distance` function."""
    if num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    if distance_metric not in ["euclidean", "chessboard", "taxicab"]:
        raise ValueError(
            f"Arg `distance_metric` must be one of 'euclidean', 'chessboard', 'taxicab', but got {distance_metric}."
        )
    if spacing is not None and not isinstance(spacing, (list, Tensor)):
        raise ValueError(f"Arg `spacing` must be a list or tensor, but got {type(spacing)}.")
    if not isinstance(directed, bool):
        raise ValueError(f"Expected argument `directed` must be a boolean, but got {directed}.")
    if input_format not in ["one-hot", "index"]:
        raise ValueError(f"Expected argument `input_format` to be one of 'one-hot', 'index', but got {input_format}.")


def hausdorff_distance(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
    directed: bool = False,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    """Calculate `Hausdorff Distance`_ for semantic segmentation.

    Args:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map
        num_classes: number of classes
        include_background: whether to include background class in calculation
        distance_metric: distance metric to calculate surface distance. Choose one of `"euclidean"`,
          `"chessboard"` or `"taxicab"`
        spacing: spacing between pixels along each spatial dimension. If not provided the spacing is assumed to be 1
        directed: whether to calculate directed or undirected Hausdorff distance
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
          or ``"index"`` for index tensors

    Returns:
        Hausdorff Distance for each class and batch element

    Example:
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import hausdorff_distance
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> hausdorff_distance(preds, target, num_classes=5)
        tensor([[2.0000, 1.4142, 2.0000, 2.0000],
                [1.4142, 2.0000, 2.0000, 2.0000],
                [2.0000, 2.0000, 1.4142, 2.0000],
                [2.0000, 2.8284, 2.0000, 2.2361]])

    """
    _hausdorff_distance_validate_args(num_classes, include_background, distance_metric, spacing, directed, input_format)
    _check_same_shape(preds, target)

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if not include_background:
        preds, target = _ignore_background(preds, target)

    distances = torch.zeros(preds.shape[0], preds.shape[1], device=preds.device)

    # TODO: add support for batched inputs
    for b in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            dist = edge_surface_distance(
                preds=preds[b, c],
                target=target[b, c],
                distance_metric=distance_metric,
                spacing=spacing,
                symmetric=not directed,
            )
            distances[b, c] = torch.max(dist) if directed else torch.max(dist[0].max(), dist[1].max())  # type: ignore
    return distances
