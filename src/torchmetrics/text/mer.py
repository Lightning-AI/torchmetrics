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

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.mer import _mer_compute, _mer_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MatchErrorRate.plot"]


class MatchErrorRate(Metric):
    r"""Match Error Rate (`MER`_) is a common metric of the performance of an automatic speech recognition system.

    This value indicates the percentage of words that were incorrectly predicted and inserted.
    The lower the value, the better the performance of the ASR system with a MatchErrorRate of 0 being a perfect score.
    Match error rate can then be computed as:

    .. math::
        mer = \frac{S + D + I}{N + I} = \frac{S + D + I}{S + D + C + I}

    where:
        - :math:`S` is the number of substitutions,
        - :math:`D` is the number of deletions,
        - :math:`I` is the number of insertions,
        - :math:`C` is the number of correct words,
        - :math:`N` is the number of words in the reference (:math:`N=S+D+C`).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): Transcription(s) to score as a string or list of strings
    - ``target`` (:class:`~List`): Reference(s) for each speech input as a string or list of strings

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mer`` (:class:`~torch.Tensor`): A tensor with the match error rate

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Examples:
        >>> from torchmetrics.text import MatchErrorRate
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> mer = MatchErrorRate()
        >>> mer(preds, target)
        tensor(0.4444)
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    errors: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(
        self,
        preds: Union[str, List[str]],
        target: Union[str, List[str]],
    ) -> None:
        """Update state with predictions and targets."""
        errors, total = _mer_update(preds, target)
        self.errors += errors
        self.total += total

    def compute(self) -> Tensor:
        """Calculate the Match error rate."""
        return _mer_compute(self.errors, self.total)

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

            >>> # Example plotting a single value
            >>> from torchmetrics.text import MatchErrorRate
            >>> metric = MatchErrorRate()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import MatchErrorRate
            >>> metric = MatchErrorRate()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
