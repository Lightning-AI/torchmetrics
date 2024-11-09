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
from typing import Any, List, Literal, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.text.edit import _edit_distance_compute, _edit_distance_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["EditDistance.plot"]


class EditDistance(Metric):
    """Calculates the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to transform the
    predicted text into the reference text. The lower the distance, the more accurate the model is considered to be.

    Implementation is similar to `nltk.edit_distance <https://www.nltk.org/_modules/nltk/metrics/distance.html>`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of hypothesis corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``eed`` (:class:`~torch.Tensor`): A tensor with the extended edit distance score. If `reduction` is set to
      ``'none'`` or ``None``, this has shape ``(N, )``, where ``N`` is the batch size. Otherwise, this is a scalar.

    Args:
        substitution_cost: The cost of substituting one character for another.
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        Basic example with two strings. Going from “rain” -> “sain” -> “shin” -> “shine” takes 3 edits:

        >>> from torchmetrics.text import EditDistance
        >>> metric = EditDistance()
        >>> metric(["rain"], ["shine"])
        tensor(3.)

    Example::
        Basic example with two strings and substitution cost of 2. Going from “rain” -> “sain” -> “shin” -> “shine”
        takes 3 edits, where two of them are substitutions:

        >>> from torchmetrics.text import EditDistance
        >>> metric = EditDistance(substitution_cost=2)
        >>> metric(["rain"], ["shine"])
        tensor(5.)

    Example::
        Multiple strings example:

        >>> from torchmetrics.text import EditDistance
        >>> metric = EditDistance(reduction=None)
        >>> metric(["rain", "lnaguaeg"], ["shine", "language"])
        tensor([3, 4], dtype=torch.int32)
        >>> metric = EditDistance(reduction="mean")
        >>> metric(["rain", "lnaguaeg"], ["shine", "language"])
        tensor(3.5000)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    edit_scores_list: List[Tensor]
    edit_scores: Tensor
    num_elements: Tensor

    def __init__(
        self, substitution_cost: int = 1, reduction: Optional[Literal["mean", "sum", "none"]] = "mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not (isinstance(substitution_cost, int) and substitution_cost >= 0):
            raise ValueError(
                f"Expected argument `substitution_cost` to be a positive integer, but got {substitution_cost}"
            )
        self.substitution_cost = substitution_cost

        allowed_reduction = (None, "mean", "sum", "none")
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction}, but got {reduction}")
        self.reduction = reduction

        if self.reduction == "none" or self.reduction is None:
            self.add_state("edit_scores_list", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("edit_scores", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        distance = _edit_distance_update(preds, target, self.substitution_cost)
        if self.reduction == "none" or self.reduction is None:
            self.edit_scores_list.append(distance)
        else:
            self.edit_scores += distance.sum()
            self.num_elements += distance.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute the edit distance over state."""
        if self.reduction == "none" or self.reduction is None:
            return _edit_distance_compute(dim_zero_cat(self.edit_scores_list), 1, self.reduction)
        return _edit_distance_compute(self.edit_scores, self.num_elements, self.reduction)

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
            >>> from torchmetrics.text import EditDistance
            >>> metric = EditDistance()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import EditDistance
            >>> metric = EditDistance()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
