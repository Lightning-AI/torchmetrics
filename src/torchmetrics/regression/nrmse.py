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

from torchmetrics.functional.regression.nrmse import (
    _mean_squared_error_update,
    _normalized_root_mean_squared_error_compute,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["NormalizedRootMeanSquaredError.plot"]


def _final_aggregation(
    min_val: Tensor,
    max_val: Tensor,
    mean_val: Tensor,
    var_val: Tensor,
    target_squared: Tensor,
    total: Tensor,
    normalization: Literal["mean", "range", "std", "l2"] = "mean",
) -> Tensor:
    """In the case of multiple devices we need to aggregate the statistics from the different devices."""
    if len(min_val) == 1:
        if normalization == "mean":
            return mean_val[0]
        if normalization == "range":
            return max_val[0] - min_val[0]
        if normalization == "std":
            return var_val[0]
        if normalization == "l2":
            return target_squared[0]

    min_val_1, max_val_1, mean_val_1, var_val_1, target_squared_1, total_1 = (
        min_val[0],
        max_val[0],
        mean_val[0],
        var_val[0],
        target_squared[0],
        total[0],
    )
    for i in range(1, len(min_val)):
        min_val_2, max_val_2, mean_val_2, var_val_2, target_squared_2, total_2 = (
            min_val[i],
            max_val[i],
            mean_val[i],
            var_val[i],
            target_squared[i],
            total[i],
        )
        # update total and mean
        total = total_1 + total_2
        mean = (total_1 * mean_val_1 + total_2 * mean_val_2) / total

        # update variance
        _temp = (total_1 + 1) * mean - total_1 * mean_val_1
        var_val_1 += (_temp - mean_val_1) * (_temp - mean) - (_temp - mean) ** 2
        _temp = (total_2 + 1) * mean - total_2 * mean_val_2
        var_val_2 += (_temp - mean_val_2) * (_temp - mean) - (_temp - mean) ** 2
        var = var_val_1 + var_val_2

        # update min and max and target squared
        min_val = torch.min(min_val_1, min_val_2)
        max_val = torch.max(max_val_1, max_val_2)
        target_squared = target_squared_1 + target_squared_2

    if normalization == "mean":
        return mean
    if normalization == "range":
        return max_val - min_val
    if normalization == "std":
        return (var / total).sqrt()
    return target_squared.sqrt()


class NormalizedRootMeanSquaredError(Metric):
    r"""Calculates the `Normalized Root Mean Squared Error`_ (NRMSE) also know as scatter index.

    The metric is defined as:

    .. math::
        \text{NRMSE} = \frac{\text{RMSE}}{\text{denom}}

    where RMSE is the root mean squared error and `denom` is the normalization factor. The normalization factor can be
    either be the mean, range, standard deviation or L2 norm of the target, which can be set using the `normalization`
    argument.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``nrmse`` (:class:`~torch.Tensor`): A tensor with the mean squared error

    Args:
        normalization: type of normalization to be applied. Choose from "mean", "range", "std", "l2" which corresponds
          to normalizing the RMSE by the mean of the target, the range of the target, the standard deviation of the
          target or the L2 norm of the target.
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        Single output normalized root mean squared error computation:

        >>> import torch
        >>> from torchmetrics import NormalizedRootMeanSquaredError
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> nrmse = NormalizedRootMeanSquaredError(normalization="mean")
        >>> nrmse(preds, target)
        tensor(0.1919)
        >>> nrmse = NormalizedRootMeanSquaredError(normalization="range")
        >>> nrmse(preds, target)
        tensor(0.1701)

    Example::
        Multioutput normalized root mean squared error computation:

        >>> import torch
        >>> from torchmetrics import NormalizedRootMeanSquaredError
        >>> preds = torch.tensor([[0., 1], [2, 3], [4, 5], [6, 7]])
        >>> target = torch.tensor([[0., 1], [3, 3], [4, 5], [8, 9]])
        >>> nrmse = NormalizedRootMeanSquaredError(num_outputs=2)
        >>> nrmse(preds, target)
        tensor([0.2981, 0.2222])

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    plot_lower_bound: float = 0.0

    sum_squared_error: Tensor
    total: Tensor
    min_val: Tensor
    max_val: Tensor
    target_squared: Tensor
    mean_val: Tensor
    var_val: Tensor

    def __init__(
        self,
        normalization: Literal["mean", "range", "std", "l2"] = "mean",
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if normalization not in ("mean", "range", "std", "l2"):
            raise ValueError(
                f"Argument `normalization` should be either 'mean', 'range', 'std' or 'l2', but got {normalization}"
            )
        self.normalization = normalization

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_outputs), dist_reduce_fx=None)
        self.add_state("min_val", default=float("Inf") * torch.ones(self.num_outputs), dist_reduce_fx=None)
        self.add_state("max_val", default=-float("Inf") * torch.ones(self.num_outputs), dist_reduce_fx=None)
        self.add_state("mean_val", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_val", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("target_squared", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        See `mean_squared_error_update` for details.

        """
        sum_squared_error, num_obs = _mean_squared_error_update(preds, target, self.num_outputs)
        self.sum_squared_error += sum_squared_error
        target = target.view(-1) if self.num_outputs == 1 else target

        # Update min and max and target squared
        self.min_val = torch.minimum(target.min(dim=0).values, self.min_val)
        self.max_val = torch.maximum(target.max(dim=0).values, self.max_val)
        self.target_squared += (target**2).sum(dim=0)

        # Update mean and variance
        new_mean = (self.total * self.mean_val + target.sum(dim=0)) / (self.total + num_obs)
        self.total += num_obs
        new_var = ((target - new_mean) * (target - self.mean_val)).sum(dim=0)
        self.mean_val = new_mean
        self.var_val += new_var

    def compute(self) -> Tensor:
        """Computes NRMSE over state.

        See `mean_squared_error_compute` for details.

        """
        if (self.num_outputs == 1 and self.mean_val.numel() > 1) or (self.num_outputs > 1 and self.mean_val.ndim > 1):
            denom = _final_aggregation(
                min_val=self.min_val,
                max_val=self.max_val,
                mean_val=self.mean_val,
                var_val=self.var_val,
                target_squared=self.target_squared,
                total=self.total,
                normalization=self.normalization,
            )
            total = self.total.squeeze().sum(dim=0)
        else:
            if self.normalization == "mean":
                denom = self.mean_val
            elif self.normalization == "range":
                denom = self.max_val - self.min_val
            elif self.normalization == "std":
                denom = torch.sqrt(self.var_val / self.total)
            else:
                denom = torch.sqrt(self.target_squared)
            total = self.total
        return _normalized_root_mean_squared_error_compute(self.sum_squared_error, total, denom)

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
            >>> from torchmetrics.regression import NormalizedRootMeanSquaredError
            >>> metric = NormalizedRootMeanSquaredError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import NormalizedRootMeanSquaredError
            >>> metric = NormalizedRootMeanSquaredError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
