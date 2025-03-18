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

from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.explained_variance import (
    ALLOWED_MULTIOUTPUT,
    _explained_variance_compute,
    _explained_variance_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ExplainedVariance.plot"]


class ExplainedVariance(Metric):
    r"""Compute `explained variance`_.

    .. math:: \text{ExplainedVariance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model in float tensor
      with shape ``(N,)`` or ``(N, ...)`` (multioutput)
    - ``target`` (:class:`~torch.Tensor`): Ground truth values in long tensor
      with shape ``(N,)`` or ``(N, ...)`` (multioutput)

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``explained_variance`` (:class:`~torch.Tensor`): A tensor with the explained variance(s)

    In the case of multioutput, as default the variances will be uniformly averaged over the additional dimensions.
    Please see argument ``multioutput`` for changing this behavior.

    Args:
        multioutput:
            Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is ``'uniform_average'``.):

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``multioutput`` is not one of ``"raw_values"``, ``"uniform_average"`` or ``"variance_weighted"``.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.regression import ExplainedVariance
        >>> target = tensor([3, -0.5, 2, 7])
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> explained_variance = ExplainedVariance()
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance = ExplainedVariance(multioutput='raw_values')
        >>> explained_variance(preds, target)
        tensor([0.9677, 1.0000])

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    num_obs: Tensor
    sum_error: Tensor
    sum_squared_error: Tensor
    sum_target: Tensor
    sum_squared_target: Tensor

    def __init__(
        self,
        multioutput: Literal["raw_values", "uniform_average", "variance_weighted"] = "uniform_average",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if multioutput not in ALLOWED_MULTIOUTPUT:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {ALLOWED_MULTIOUTPUT}"
            )
        self.multioutput = multioutput
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_target", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_target", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(
            preds, target
        )
        self.num_obs = self.num_obs + num_obs
        self.sum_error = self.sum_error + sum_error
        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.sum_target = self.sum_target + sum_target
        self.sum_squared_target = self.sum_squared_target + sum_squared_target

    def compute(self) -> Union[Tensor, Sequence[Tensor]]:
        """Compute explained variance over state."""
        return _explained_variance_compute(
            self.num_obs,
            self.sum_error,
            self.sum_squared_error,
            self.sum_target,
            self.sum_squared_target,
            self.multioutput,
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
            >>> from torchmetrics.regression import ExplainedVariance
            >>> metric = ExplainedVariance()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import ExplainedVariance
            >>> metric = ExplainedVariance()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
