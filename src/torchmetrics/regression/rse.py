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
from torch import Tensor, tensor

from torchmetrics.functional.regression.r2 import _r2_score_update
from torchmetrics.functional.regression.rse import _relative_squared_error_compute
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["RelativeSquaredError.plot"]


class RelativeSquaredError(Metric):
    r"""Computes the relative squared error (RSE).

    .. math:: \text{RSE} = \frac{\sum_i^N(y_i - \hat{y_i})^2}{\sum_i^N(y_i - \overline{y})^2}

    Where :math:`y` is a tensor of target values with mean :math:`\overline{y}`, and
    :math:`\hat{y}` is a tensor of predictions.

    If num_outputs > 1, the returned value is averaged over all the outputs.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model in float tensor with shape ``(N,)``
      or ``(N, M)`` (multioutput)
    - ``target`` (:class:`~torch.Tensor`): Ground truth values in float tensor with shape ``(N,)``
      or ``(N, M)`` (multioutput)

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``rse`` (:class:`~torch.Tensor`): A tensor with the RSE score(s)

    Args:
        num_outputs: Number of outputs in multioutput setting
        squared: If True returns RSE value, if False returns RRSE value.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.regression import RelativeSquaredError
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> relative_squared_error = RelativeSquaredError()
        >>> relative_squared_error(preds, target)
        tensor(0.0514)

    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    sum_error: Tensor
    residual: Tensor
    total: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("residual", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, sum_error, residual, total = _r2_score_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total

    def compute(self) -> Tensor:
        """Computes relative squared error over state."""
        return _relative_squared_error_compute(
            self.sum_squared_error, self.sum_error, self.residual, self.total, squared=self.squared
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

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from torchmetrics.regression import RelativeSquaredError
            >>> metric = RelativeSquaredError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import RelativeSquaredError
            >>> metric = RelativeSquaredError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
