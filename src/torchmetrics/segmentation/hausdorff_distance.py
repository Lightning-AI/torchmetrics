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
from typing import Any, List, Literal, Optional, Sequence, Union

from torch import Tensor

from torchmetrics.functional.segmentation import hausdorff_distance
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat, dim_zero_mean
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["HausdorffDistance.plot"]


class HausdorffDistance(Metric):
    r"""Compute the Hausdorff distance between two subsets of a metric space.

    .. math::
        d_{\Pi}(X,Y) = \max{/sup_{x\in X} {d(x,Y)}, /sup_{y\in Y} {d(X,y)}}

    where :math:`\X, \Y` are ________________, :math:`\X, \Y` ______.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`):
    - ``target`` (:class:`~torch.Tensor`):

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``hausdorff_distance`` (:class:`~torch.Tensor`): A scalar float tensor with the Hausdorff distance.

    Args:
        p: p-norm used for distance metric
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
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
        >>> hausdorff_distance = HausdorffDistance(distance_metric="euclidean")
        >>> hausdorff_distance.update(preds, target)
        >>> hausdorff_distance.compute()
        tensor(1.)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
        spacing: Optional[Union[Tensor, List[float]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.distance_metric = distance_metric
        self.spacing = spacing

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute final Hausdorff distance over states."""
        return dim_zero_mean(
            dim_zero_cat([
                hausdorff_distance(p, t, self.distance_metric, self.spacing) for p, t in zip(self.preds, self.target)
            ])
        )

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
            >>> metric = HausdorffDistance()
            >>> data1, data2 = randint(0, 2, (1,10,)).bool(), randint(0, 2, (1,10,)).bool()
            >>> metric.update(data1, data2)
            >>> fig_, ax_ = metric.plot()

        """
        return self._plot(val, ax)
