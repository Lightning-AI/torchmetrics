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
    """Calculates the edit distance between two sequences.

    Args:
        substitution_cost: The cost of substituting one character for another.
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    """

    def __init__(
        self, substitution_cost: int = 1, reduction: Optional[Literal["mean", "sum", "none"]] = "mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not (isinstance(substitution_cost) and substitution_cost >= 0):
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
