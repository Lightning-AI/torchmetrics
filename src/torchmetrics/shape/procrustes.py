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
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.shape.procrustes import procrustes_disparity
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ProcrustesDisparity.plot"]


class ProcrustesDisparity(Metric):
    r"""Compute the `Procrustes Disparity`_.

    The Procrustes Disparity is defined as the sum of the squared differences between two datasets after
    applying a Procrustes transformation. The Procrustes Disparity is useful to compare two datasets
    that are similar but not aligned.

    The metric works similar to ``scipy.spatial.procrustes`` but for batches of data points. The disparity is
    aggregated over the batch, thus to get the individual disparities please use the functional version of this
    metric: ``torchmetrics.functional.shape.procrustes.procrustes_disparity``.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``point_cloud1`` (torch.Tensor): A tensor of shape ``(N, M, D)`` with ``N`` being the batch size,
          ``M`` the number of data points and ``D`` the dimensionality of the data points.
        - ``point_cloud2`` (torch.Tensor): A tensor of shape ``(N, M, D)`` with ``N`` being the batch size,
          ``M`` the number of data points and ``D`` the dimensionality of the data points.


    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``gds`` (:class:`~torch.Tensor`): A scalar tensor with the Procrustes Disparity.

    Args:
        reduction: Determines whether to return the mean disparity or the sum of the disparities.
            Can be one of ``"mean"`` or ``"sum"``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError: If ``average`` is not one of ``"mean"`` or ``"sum"``.

    Example:
        >>> from torch import randn
        >>> from torchmetrics.shape import ProcrustesDisparity
        >>> metric = ProcrustesDisparity()
        >>> point_cloud1 = randn(10, 50, 2)
        >>> point_cloud2 = randn(10, 50, 2)
        >>> metric(point_cloud1, point_cloud2)
        tensor(0.9770)

    """

    disparity: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, reduction: Literal["mean", "sum"] = "mean", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Argument `reduction` must be one of ['mean', 'sum'], got {reduction}")
        self.reduction = reduction
        self.add_state("disparity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, point_cloud1: torch.Tensor, point_cloud2: torch.Tensor) -> None:
        """Update the Procrustes Disparity with the given datasets."""
        disparity: Tensor = procrustes_disparity(point_cloud1, point_cloud2)  # type: ignore[assignment]
        self.disparity += disparity.sum()
        self.total += disparity.numel()

    def compute(self) -> torch.Tensor:
        """Computes the Procrustes Disparity."""
        if self.reduction == "mean":
            return self.disparity / self.total
        return self.disparity

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
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

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.shape import ProcrustesDisparity
            >>> metric = ProcrustesDisparity()
            >>> metric.update(torch.randn(10, 50, 2), torch.randn(10, 50, 2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.shape import ProcrustesDisparity
            >>> metric = ProcrustesDisparity()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randn(10, 50, 2), torch.randn(10, 50, 2)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
