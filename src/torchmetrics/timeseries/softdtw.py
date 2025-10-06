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
from typing import Any, Optional, Union, Callable, List

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.timeseries.softdtw import soft_dtw
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SoftDTW.plot"]


class SoftDTW(Metric):
    r"""Compute the **Soft Dynamic Time Warping (Soft-DTW)** distance between two batched sequences.

    Compute the **Soft Dynamic Time Warping (Soft-DTW)** distance between two batched sequences.

    This is a differentiable relaxation of the classic Dynamic Time Warping (DTW) algorithm, introduced by
    Marco Cuturi and Mathieu Blondel (2017).
    It replaces the hard minimum in DTW recursion with a *soft-minimum* using a log-sum-exp formulation:

    .. math::
        \text{softmin}_\gamma(a,b,c) = -\gamma \log \left( e^{-a/\gamma} + e^{-b/\gamma} + e^{-c/\gamma} \right)

    The Soft-DTW recurrence is then defined as:

    .. math::
        R_{i,j} = D_{i,j} + \text{softmin}_\gamma(R_{i-1,j}, R_{i,j-1}, R_{i-1,j-1})

    where :math:`D_{i,j}` is the pairwise distance between sequence elements :math:`x_i` and :math:`y_j`.

    The final Soft-DTW distance is :math:`R_{N,M}`.

    Args:
        gamma: Smoothing parameter (:math:`\gamma > 0`). Smaller values make the loss closer to standard DTW (hard minimum),
            while larger values produce a smoother and more differentiable surface.
        distance_fn: Optional callable ``(x, y) -> [B, N, M]`` defining the pairwise distance matrix.
            If ``None``, defaults to **squared Euclidean distance**.

    Raises:
        ValueError:
            If ``gamma`` is not a positive float.
            If input tensors to ``update`` are not 3-dimensional with the same batch size and feature dimension.

    Example:
        >>> from torch import randn
        >>> from torchmetrics.timeseries import SoftDTW
        >>> metric = SoftDTW(gamma=0.1)
        >>> x = randn(10, 50, 2)
        >>> y = randn(10, 60, 2)
        >>> metric(x, y)
        tensor(43.2051)
    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_list: List[Tensor]
    gt_list: List[Tensor]

    def __init__(self, 
                distance_fn: Optional[Callable] = None,
                gamma: float = 1.0,
                **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.distance_fn = distance_fn
        if gamma <= 0:
            raise ValueError(f"Argument `gamma` must be a positive float, got {gamma}")
        self.gamma = gamma

        self.add_state("pred_list", default=[], dist_reduce_fx="cat")
        self.add_state("gt_list", default=[], dist_reduce_fx="cat")

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Update the Procrustes Disparity with the given datasets."""
        self.pred_list.append(x)
        self.gt_list.append(y)

    def compute(self) -> torch.Tensor:
        """Computes the Procrustes Disparity."""
        return soft_dtw(
            torch.cat(self.pred_list, dim=0),
            torch.cat(self.gt_list, dim=0),
            gamma=self.gamma,
            distance_fn=self.distance_fn,
        ).mean()

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
            >>> from torchmetrics.timeseries import SoftDTW
            >>> metric = SoftDTW()
            >>> metric.update(torch.randn(10, 100, 2), torch.randn(10, 50, 2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.timeseries import SoftDTW
            >>> metric = SoftDTW()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randn(10, 100, 2), torch.randn(10, 50, 2)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
