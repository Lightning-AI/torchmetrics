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
from typing import Any, Optional, Sequence, Union

from torch import Tensor, tensor

from torchmetrics.functional.regression.symmetric_mape import (
    _symmetric_mean_absolute_percentage_error_compute,
    _symmetric_mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SymmetricMeanAbsolutePercentageError.plot"]


class SymmetricMeanAbsolutePercentageError(Metric):
    r"""Compute symmetric mean absolute percentage error (`SMAPE`_).

    .. math:: \text{SMAPE} = \frac{2}{n}\sum_1^n\frac{|   y_i - \hat{y_i} |}{\max(| y_i | + | \hat{y_i} |, \epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``smape`` (:class:`~torch.Tensor`): A tensor with non-negative floating point smape value between 0 and 1

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
        >>> target = tensor([1, 10, 1e6])
        >>> preds = tensor([0.9, 15, 1.2e6])
        >>> smape = SymmetricMeanAbsolutePercentageError()
        >>> smape(preds, target)
        tensor(0.2290)
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute percentage error over state."""
        return _symmetric_mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

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
            >>> from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
            >>> metric = SymmetricMeanAbsolutePercentageError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
            >>> metric = SymmetricMeanAbsolutePercentageError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)
        """
        return self._plot(val, ax)
