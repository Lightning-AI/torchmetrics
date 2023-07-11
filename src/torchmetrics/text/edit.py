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
from typing import Any, List, Literal, Optional

import torch

from torchmetrics.functional.text.edit import _edit_distance_compute, _edit_distance_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class EditDistance(Metric):
    """Calculates the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to transform the
    predicted text into the reference text. The lower the distance, the more accurate the model is considered to be.

    Implementation is similar to `nltk.edit_distance <https://www.nltk.org/_modules/nltk/metrics/distance.html>`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): An iterable of hypothesis corpus
    - ``target`` (:class:`~List`): An iterable of iterables of reference corpus

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

    def __init__(
        self, substitution_cost: int = 1, reduction: Optional[Literal["mean", "sum", "none"]] = "mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not (isinstance(substitution_cost, int) and substitution_cost >= 0):
            raise ValueError("Expected argument `substitution_cost` to be a positive integer")
        self.substitution_cost = substitution_cost

        allowed_reduction = (None, "mean", "sum", "none")
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction}")
        self.reduction = reduction

        if self.reduction == "none" or self.reduction is None:
            self.add_state("edit_scores_list", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("edit_scores", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[str], target: List[str]) -> None:
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
