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
from typing import Optional, Sequence, Union

from torch import Tensor

from torchmetrics.functional.regression.concordance import _concordance_corrcoef_compute
from torchmetrics.regression.pearson import PearsonCorrCoef, _final_aggregation
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ConcordanceCorrCoef.plot"]


class ConcordanceCorrCoef(PearsonCorrCoef):
    r"""Compute concordance correlation coefficient that measures the agreement between two variables.

    .. math::
        \rho_c = \frac{2 \rho \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}

    where :math:`\mu_x, \mu_y` is the means for the two variables, :math:`\sigma_x^2, \sigma_y^2` are the corresponding
    variances and \rho is the pearson correlation coefficient between the two variables.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)`` or multioutput
      float tensor of shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)`` or multioutput
      float tensor of shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``concordance`` (:class:`~torch.Tensor`): A scalar float tensor with the concordance coefficient(s) for
      non-multioutput input or a float tensor with shape ``(d,)`` for multioutput input

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torchmetrics.regression import ConcordanceCorrCoef
        >>> from torch import tensor
        >>> target = tensor([3, -0.5, 2, 7])
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> concordance = ConcordanceCorrCoef()
        >>> concordance(preds, target)
        tensor(0.9777)

    Example (multi output regression):
        >>> from torchmetrics.regression import ConcordanceCorrCoef
        >>> target = tensor([[3, -0.5], [2, 7]])
        >>> preds = tensor([[2.5, 0.0], [2, 8]])
        >>> concordance = ConcordanceCorrCoef(num_outputs=2)
        >>> concordance(preds, target)
        tensor([0.7273, 0.9887])
    """
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute final concordance correlation coefficient over metric states."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            mean_x, mean_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
            )
        else:
            mean_x = self.mean_x
            mean_y = self.mean_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _concordance_corrcoef_compute(mean_x, mean_y, var_x, var_y, corr_xy, n_total)

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
            >>> from torchmetrics.regression import ConcordanceCorrCoef
            >>> metric = ConcordanceCorrCoef()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import ConcordanceCorrCoef
            >>> metric = ConcordanceCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)
        """
        return self._plot(val, ax)
