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
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from torchmetrics.functional.regression.spearman import _spearman_corrcoef_compute, _spearman_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpearmanCorrCoef.plot"]


class SpearmanCorrCoef(Metric):
    r"""Compute `spearmans rank correlation coefficient`_.

    .. math:
        r_s = = \frac{cov(rg_x, rg_y)}{\sigma_{rg_x} * \sigma_{rg_y}}

    where :math:`rg_x` and :math:`rg_y` are the rank associated to the variables :math:`x` and :math:`y`.
    Spearmans correlations coefficient corresponds to the standard pearsons correlation coefficient calculated
    on the rank variables.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model in float tensor with shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values in float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``spearman`` (:class:`~torch.Tensor`): A tensor with the spearman correlation(s)

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torch import tensor
        >>> from torchmetrics.regression import SpearmanCorrCoef
        >>> target = tensor([3, -0.5, 2, 7])
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> spearman = SpearmanCorrCoef()
        >>> spearman(preds, target)
        tensor(1.0000)

    Example (multi output regression):
        >>> from torchmetrics.regression import SpearmanCorrCoef
        >>> target = tensor([[3, -0.5], [2, 7]])
        >>> preds = tensor([[2.5, 0.0], [2, 8]])
        >>> spearman = SpearmanCorrCoef(num_outputs=2)
        >>> spearman(preds, target)
        tensor([1.0000, 1.0000])
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError("Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _spearman_corrcoef_update(preds, target, num_outputs=self.num_outputs)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute Spearman's correlation coefficient."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spearman_corrcoef_compute(preds, target)

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
            >>> from torchmetrics.regression import SpearmanCorrCoef
            >>> metric = SpearmanCorrCoef()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import SpearmanCorrCoef
            >>> metric = SpearmanCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)
        """
        return self._plot(val, ax)
