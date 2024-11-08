# Copyright The Lightning team.
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
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.segmentation.hausdorff_distance import (
    _hausdorff_distance_validate_args,
    hausdorff_distance,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["HausdorffDistance.plot"]


class HausdorffDistance(Metric):
    r"""Compute the `Hausdorff Distance`_ between two subsets of a metric space for semantic segmentation.

    .. math::
        d_{\Pi}(X,Y) = \max{/sup_{x\in X} {d(x,Y)}, /sup_{y\in Y} {d(X,y)}}

    where :math:`\X, \Y` are two subsets of a metric space with distance metric :math:`d`. The Hausdorff distance is
    the maximum distance from a point in one set to the closest point in the other set. The Hausdorff distance is a
    measure of the degree of mismatch between two sets.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
        the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
        can be provided, where the integer values correspond to the class index. The input type can be controlled
        with the ``input_format`` argument.
    - ``target`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
        the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
        can be provided, where the integer values correspond to the class index. The input type can be controlled
        with the ``input_format`` argument.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``hausdorff_distance`` (:class:`~torch.Tensor`): A scalar float tensor with the Hausdorff distance averaged over
        classes and samples

    Args:
        num_classes: number of classes
        include_background: whether to include background class in calculation
        distance_metric: distance metric to calculate surface distance. Choose one of `"euclidean"`,
          `"chessboard"` or `"taxicab"`
        spacing: spacing between pixels along each spatial dimension. If not provided the spacing is assumed to be 1
        directed: whether to calculate directed or undirected Hausdorff distance
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
          or ``"index"`` for index tensors
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import randint
        >>> from torchmetrics.segmentation import HausdorffDistance
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> hausdorff_distance = HausdorffDistance(distance_metric="euclidean", num_classes=5)
        >>> hausdorff_distance(preds, target)
        tensor(1.9567)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    score: Tensor
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
        spacing: Optional[Union[Tensor, list[float]]] = None,
        directed: bool = False,
        input_format: Literal["one-hot", "index"] = "one-hot",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _hausdorff_distance_validate_args(
            num_classes, include_background, distance_metric, spacing, directed, input_format
        )
        self.num_classes = num_classes
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.spacing = spacing
        self.directed = directed
        self.input_format = input_format
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        score = hausdorff_distance(
            preds,
            target,
            self.num_classes,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            spacing=self.spacing,
            directed=self.directed,
            input_format=self.input_format,
        )
        self.score += score.sum()
        self.total += score.numel()

    def compute(self) -> Tensor:
        """Compute final Hausdorff distance over states."""
        return self.score / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import randint
            >>> from torchmetrics.segmentation import HausdorffDistance
            >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
            >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
            >>> metric = HausdorffDistance(num_classes=5)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        """
        return self._plot(val, ax)
