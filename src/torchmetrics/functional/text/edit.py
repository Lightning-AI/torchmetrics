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
from typing import List, Literal, Optional, Union

import torch

from torchmetrics.functional.text.helper import _LevenshteinEditDistance


def _edit_distance_update(
    preds: List[str],
    target: List[str],
    substitution_cost: int = 1,
) -> torch.Tensor:
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(target, list):
        target = [target]
    if not all(isinstance(x, str) for x in preds):
        raise ValueError("Expected all values in argument `preds` to be string type")
    if not all(isinstance(x, str) for x in target):
        raise ValueError("Expected all values in argument `target` to be string type")
    if len(preds) != len(target):
        raise ValueError("Expected argument `preds` and `target` to have same length")

    distance = [_LevenshteinEditDistance(t, op_substitute=substitution_cost)(p)[0] for p, t in zip(preds, target)]
    return torch.tensor(distance, dtype=torch.int)


def _edit_distance_compute(
    edit_scores: torch.Tensor,
    num_elements: Union[torch.Tensor, int],
    reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
) -> torch.Tensor:
    """Compute final edit distance reduced over the batch."""
    if reduction == "mean":
        return edit_scores.sum() / num_elements
    if reduction == "sum":
        return edit_scores.sum()
    if reduction is None or reduction == "none":
        return edit_scores
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


def edit_distance(
    preds: List[str],
    target: List[str],
    substitution_cost: int = 1,
    reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
) -> int:
    """https://www.nltk.org/api/nltk.metrics.distance.html#module-nltk.metrics.distance."""
    distance = _edit_distance_update(preds, target, substitution_cost)
    return _edit_distance_compute(distance, num_elements=distance.numel(), reduction=reduction)
