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
from typing_extensions import Literal

from torchmetrics.functional.nominal.fleiss_kappa import _fleiss_kappa_compute, _fleiss_kappa_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FleissKappa.plot"]


class FleissKappa(Metric):
    r"""Calculatees `Fleiss kappa`_ a statistical measure for inter agreement between raters.

    .. math::
        \kappa = \frac{\bar{p} - \bar{p_e}}{1 - \bar{p_e}}

    where :math:`\bar{p}` is the mean of the agreement probability over all raters and :math:`\bar{p_e}` is the mean
    agreement probability over all raters if they were randomly assigned. If the raters are in complete agreement then
    the score 1 is returned, if there is no agreement among the raters (other than what would be expected by chance)
    then a score smaller than 0 is returned.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``ratings`` (:class:`~torch.Tensor`): Ratings of shape ``[n_samples, n_categories]`` or
      ``[n_samples, n_categories, n_raters]`` depedenent on ``mode``. If ``mode`` is ``counts``, ``ratings`` must be
      integer and contain the number of raters that chose each category. If ``mode`` is ``probs``, ``ratings`` must be
      floating point and contain the probability/logits that each rater chose each category.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``fleiss_k`` (:class:`~torch.Tensor`): A float scalar tensor with the calculated Fleiss' kappa score.

    Args:
        mode: Whether `ratings` will be provided as counts or probabilities.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> # Ratings are provided as counts
        >>> import torch
        >>> from torchmetrics.nominal import FleissKappa
        >>> _ = torch.manual_seed(42)
        >>> ratings = torch.randint(0, 10, size=(100, 5)).long()  # 100 samples, 5 categories, 10 raters
        >>> metric = FleissKappa(mode='counts')
        >>> metric(ratings)
        tensor(0.0089)

    Example:
        >>> # Ratings are provided as probabilities
        >>> import torch
        >>> from torchmetrics.nominal import FleissKappa
        >>> _ = torch.manual_seed(42)
        >>> ratings = torch.randn(100, 5, 10).softmax(dim=1)  # 100 samples, 5 categories, 10 raters
        >>> metric = FleissKappa(mode='probs')
        >>> metric(ratings)
        tensor(-0.0105)

    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_upper_bound: float = 1.0
    counts: List[Tensor]

    def __init__(self, mode: Literal["counts", "probs"] = "counts", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if mode not in ["counts", "probs"]:
            raise ValueError("Argument ``mode`` must be one of 'counts' or 'probs'.")
        self.mode = mode
        self.add_state("counts", default=[], dist_reduce_fx="cat")

    def update(self, ratings: Tensor) -> None:
        """Updates the counts for fleiss kappa metric."""
        counts = _fleiss_kappa_update(ratings, self.mode)
        self.counts.append(counts)

    def compute(self) -> Tensor:
        """Computes Fleiss' kappa."""
        counts = dim_zero_cat(self.counts)
        return _fleiss_kappa_compute(counts)

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
            >>> from torchmetrics.nominal import FleissKappa
            >>> metric = FleissKappa(mode="probs")
            >>> metric.update(torch.randn(100, 5, 10).softmax(dim=1))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.nominal import FleissKappa
            >>> metric = FleissKappa(mode="probs")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randn(100, 5, 10).softmax(dim=1)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
