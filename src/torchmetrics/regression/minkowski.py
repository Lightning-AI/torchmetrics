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

from typing import Any, Optional, Sequence, Union

from torch import Tensor, tensor

from torchmetrics.functional.regression.minkowski import _minkowski_distance_compute, _minkowski_distance_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MinkowskiDistance.plot"]


class MinkowskiDistance(Metric):
    r"""Compute `Minkowski Distance`_.

    .. math:: d_{\text{Minkowski}} = \\sum_{i}^N (| y_i - \\hat{y_i} |^p)^\frac{1}{p}

    This metric can be seen as generalized version of the standard euclidean distance which corresponds to minkowski
    distance with p=2.

    where
        :math:`y` is a tensor of target values,
        :math:`\\hat{y}` is a tensor of predictions,
        :math: `\\p` is a non-negative integer or floating-point number

    Args:
        p: int or float larger than 1, exponent to which the difference between preds and target is to be raised
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.regression import MinkowskiDistance
        >>> target = tensor([1.0, 2.8, 3.5, 4.5])
        >>> preds = tensor([6.1, 2.11, 3.1, 5.6])
        >>> minkowski_distance = MinkowskiDistance(3)
        >>> minkowski_distance(preds, target)
        tensor(5.1220)
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0

    minkowski_dist_sum: Tensor

    def __init__(self, p: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not (isinstance(p, (float, int)) and p >= 1):
            raise TorchMetricsUserError(f"Argument ``p`` must be a float or int greater than 1, but got {p}")

        self.p = p
        self.add_state("minkowski_dist_sum", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update state with predictions and targets."""
        minkowski_dist_sum = _minkowski_distance_update(preds, targets, self.p)
        self.minkowski_dist_sum += minkowski_dist_sum

    def compute(self) -> Tensor:
        """Compute metric."""
        return _minkowski_distance_compute(self.minkowski_dist_sum, self.p)

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

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from torchmetrics.regression import MinkowskiDistance
            >>> metric = MinkowskiDistance(p=3)
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import MinkowskiDistance
            >>> metric = MinkowskiDistance(p=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)
        """
        return self._plot(val, ax)
