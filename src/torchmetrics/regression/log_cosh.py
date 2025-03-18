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

from torchmetrics.functional.regression.log_cosh import _log_cosh_error_compute, _log_cosh_error_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LogCoshError.plot"]


class LogCoshError(Metric):
    r"""Compute the `LogCosh Error`_.

    .. math:: \text{LogCoshError} = \log\left(\frac{\exp(\hat{y} - y) + \exp(\hat{y - y})}{2}\right)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Estimated labels with shape ``(batch_size,)``
      or ``(batch_size, num_outputs)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth labels with shape ``(batch_size,)``
      or ``(batch_size, num_outputs)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``log_cosh_error`` (:class:`~torch.Tensor`): A tensor with the log cosh error

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression)::
        >>> from torchmetrics.regression import LogCoshError
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> log_cosh_error = LogCoshError()
        >>> log_cosh_error(preds, target)
        tensor(0.3523)

    Example (multi output regression)::
        >>> from torchmetrics.regression import LogCoshError
        >>> preds = torch.tensor([[3.0, 5.0, 1.2], [-2.1, 2.5, 7.0]])
        >>> target = torch.tensor([[2.5, 5.0, 1.3], [0.3, 4.0, 8.0]])
        >>> log_cosh_error = LogCoshError(num_outputs=3)
        >>> log_cosh_error(preds, target)
        tensor([0.9176, 0.4277, 0.2194])

    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_log_cosh_error: Tensor
    total: Tensor

    def __init__(self, num_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(f"Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs
        self.add_state("sum_log_cosh_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Raises:
            ValueError:
                If ``preds`` or ``target`` has multiple outputs when ``num_outputs=1``

        """
        sum_log_cosh_error, num_obs = _log_cosh_error_update(preds, target, self.num_outputs)
        self.sum_log_cosh_error += sum_log_cosh_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute LogCosh error over state."""
        return _log_cosh_error_compute(self.sum_log_cosh_error, self.total)

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
            >>> from torchmetrics.regression import LogCoshError
            >>> metric = LogCoshError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import LogCoshError
            >>> metric = LogCoshError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
