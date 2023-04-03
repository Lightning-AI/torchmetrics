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

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.nominal.theils_u import _theils_u_compute, _theils_u_update
from torchmetrics.functional.nominal.utils import _nominal_input_validation
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TheilsU.plot"]


class TheilsU(Metric):
    r"""Compute `Theil's U`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        U(X|Y) = \frac{H(X) - H(X|Y)}{H(X)}

    where :math:`H(X)` is entropy of variable :math:`X` while :math:`H(X|Y)` is the conditional entropy of :math:`X`
    given :math:`Y`. It is also know as the Uncertainty Coefficient.

    Theils's U is an asymmetric coefficient, i.e. :math:`TheilsU(preds, target) \neq TheilsU(target, preds)`.

    The output values lies in [0, 1]. 0 means y has no information about x while value 1 means y has complete
    information about x.

    Args:
        num_classes: Integer specifing the number of classes
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Theil's U Statistic: Tensor

    Example:
        >>> from torchmetrics.nominal import TheilsU
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> metric = TheilsU(num_classes=10)
        >>> metric(preds, target)
        tensor(0.8530)
    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        nan_strategy: Literal["replace", "drop"] = "replace",
        nan_replace_value: Optional[Union[int, float]] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes

        _nominal_input_validation(nan_strategy, nan_replace_value)
        self.nan_strategy = nan_strategy
        self.nan_replace_value = nan_replace_value

        self.add_state("confmat", torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
            target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        """
        confmat = _theils_u_update(preds, target, self.num_classes, self.nan_strategy, self.nan_replace_value)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Compute Theil's U statistic."""
        return _theils_u_compute(self.confmat)

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
            >>> from torchmetrics.nominal import TheilsU
            >>> metric = TheilsU(num_classes=10)
            >>> metric.update(torch.randint(10, (10,)), torch.randint(10, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.nominal import TheilsU
            >>> metric = TheilsU(num_classes=10)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(10, (10,)), torch.randint(10, (10,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
