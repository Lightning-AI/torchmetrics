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

from torchmetrics.functional.nominal.cramers import _cramers_v_compute, _cramers_v_update
from torchmetrics.functional.nominal.utils import _nominal_input_validation
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CramersV.plot"]


class CramersV(Metric):
    r"""Compute `Cramer's V`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        V = \sqrt{\frac{\chi^2 / n}{\min(r - 1, k - 1)}}

    where

    .. math::
        \chi^2 = \sum_{i,j} \ frac{\left(n_{ij} - \frac{n_{i.} n_{.j}}{n}\right)^2}{\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively.

    Cramer's V is a symmetric coefficient, i.e. :math:`V(preds, target) = V(target, preds)`.

    The output values lies in [0, 1] with 1 meaning the perfect association.

    Args:
        num_classes: Integer specifing the number of classes
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Cramer's V statistic

    Raises:
        ValueError:
            If `nan_strategy` is not one of `'replace'` and `'drop'`
        ValueError:
            If `nan_strategy` is equal to `'replace'` and `nan_replace_value` is not an `int` or `float`

    Example:
        >>> from torchmetrics.nominal import CramersV
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(0, 4, (100,))
        >>> target = torch.round(preds + torch.randn(100)).clamp(0, 4)
        >>> cramers_v = CramersV(num_classes=5)
        >>> cramers_v(preds, target)
        tensor(0.5284)
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
        bias_correction: bool = True,
        nan_strategy: Literal["replace", "drop"] = "replace",
        nan_replace_value: Optional[Union[int, float]] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bias_correction = bias_correction

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
        confmat = _cramers_v_update(preds, target, self.num_classes, self.nan_strategy, self.nan_replace_value)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Compute Cramer's V statistic."""
        return _cramers_v_compute(self.confmat, self.bias_correction)

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
            >>> from torchmetrics.nominal import CramersV
            >>> metric = CramersV(num_classes=5)
            >>> metric.update(torch.randint(0, 4, (100,)), torch.randint(0, 4, (100,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.nominal import CramersV
            >>> metric = CramersV(num_classes=5)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(0, 4, (100,)), torch.randint(0, 4, (100,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
