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
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.regression.weighted_pearson import (
    _weighted_pearson_corrcoef_compute,
    _weighted_pearson_corrcoef_update,
)
from torchmetrics.metric import Metric
from torchmetrics.regression.pearson import _final_aggregation
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["WeightedPearsonCorrCoef.plot"]


class WeightedPearsonCorrCoef(Metric):
    r"""Compute `Weighted Pearson Correlation Coefficient`_.

    .. math::
        P_{corr}(x,y;w) = \frac{cov(x,y;w)}{cov(x,x;w) cov(y,y;w)},

    where :math:`cov(x,y;w)` is the weighted covariance,
    :math:`cov(x,x;w)` is the weighted variance of :math:`x`,
    :math:`cov(y,y;w)` is the weighted variance of :math:`y`,
    :math:`y` is a tensor of target values,
    :math:`x` is a tensor of predictions,
    and :math:`w` is a tensor of weights.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)``
      or multioutput float tensor of shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): either single output tensor with shape ``(N,)``
      or multioutput tensor of shape ``(N,d)``
    - ``weights`` (:class:`~torch.Tensor`): single tensor with shape ``(N,)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``pearson`` (:class:`~torch.Tensor`): A tensor with the weighted Pearson Correlation Coefficient

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output weighted regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> weights = torch.tensor([0.1, 0.2, 0.5])
        >>> pearson = PearsonCorrCoef()
        >>> pearson(preds, target, weights)
        tensor(0.9849)

    Example (multi output weighted regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([[3, -0.5], [2, 7], [-1, 1.5]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8], [0.0, 1.3]])
        >>> weights = torch.tensor([0.3, 0.2, 0.5])
        >>> pearson = PearsonCorrCoef(num_outputs=2)
        >>> pearson(preds, target, weights)
        tensor([1., 1., 0.1])

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None  # both -1 and 1 are optimal
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]
    mean_x: Tensor
    mean_y: Tensor
    var_x: Tensor
    var_y: Tensor
    cov_xy: Tensor
    weights_sum: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_outputs < 1:
            raise ValueError("Expected argument `num_outputs` to be an `int` larger than 0, but got {num_outputs}.")
        self.num_outputs = num_outputs

        self.add_state("mean_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("mean_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("cov_xy", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("weights_sum", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, weights: Tensor) -> None:
        """Update state with predictions and targets."""
        self.mean_x, self.mean_y, self.var_x, self.var_y, self.cov_xy, self.weights_sum = (
            _weighted_pearson_corrcoef_update(
                preds,
                target,
                weights,
                self.mean_x,
                self.mean_y,
                self.var_x,
                self.var_y,
                self.cov_xy,
                self.weights_sum,
                self.num_outputs,
            )
        )

    def compute(self) -> Tensor:
        """Compute weighted Pearson correlation coefficient over state."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            # multiple devices, need further reduction
            _, _, var_x, var_y, cov_xy, weights_sum = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.cov_xy, self.weights_sum
            )
        else:
            var_x = self.var_x
            var_y = self.var_y
            cov_xy = self.cov_xy
            weights_sum = self.weights_sum

        return _weighted_pearson_corrcoef_compute(var_x, var_y, cov_xy, weights_sum)

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
            >>> from torchmetrics.regression import WeightedPearsonCorrCoef
            >>> metric = WeightedPearsonCorrCoef()
            >>> metric.update(randn(10,), randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import WeightedPearsonCorrCoef
            >>> metric = WeightedPearsonCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
