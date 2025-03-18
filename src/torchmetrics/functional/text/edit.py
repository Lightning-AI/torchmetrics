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
from typing import Literal, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance


def _edit_distance_update(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    substitution_cost: int = 1,
) -> Tensor:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    if not all(isinstance(x, str) for x in preds):
        raise ValueError(f"Expected all values in argument `preds` to be string type, but got {preds}")
    if not all(isinstance(x, str) for x in target):
        raise ValueError(f"Expected all values in argument `target` to be string type, but got {target}")
    if len(preds) != len(target):
        raise ValueError(
            f"Expected argument `preds` and `target` to have same length, but got {len(preds)} and {len(target)}"
        )

    distance = [
        _LE_distance(t, op_substitute=substitution_cost)(p)[0]  # type: ignore[arg-type]
        for p, t in zip(preds, target)
    ]
    return torch.tensor(distance, dtype=torch.int)


def _edit_distance_compute(
    edit_scores: Tensor,
    num_elements: Union[Tensor, int],
    reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
) -> Tensor:
    """Compute final edit distance reduced over the batch."""
    if edit_scores.numel() == 0:
        return torch.tensor(0, dtype=torch.int32)
    if reduction == "mean":
        return edit_scores.sum() / num_elements
    if reduction == "sum":
        return edit_scores.sum()
    if reduction is None or reduction == "none":
        return edit_scores
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


def edit_distance(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    substitution_cost: int = 1,
    reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
) -> Tensor:
    """Calculates the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to transform the
    predicted text into the reference text. The lower the distance, the more accurate the model is considered to be.

    Implementation is similar to `nltk.edit_distance <https://www.nltk.org/_modules/nltk/metrics/distance.html>`_.

    Args:
        preds: An iterable of predicted texts (strings).
        target: An iterable of reference texts (strings).
        substitution_cost: The cost of substituting one character for another.
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Raises:
        ValueError:
            If ``preds`` and ``target`` do not have the same length.
        ValueError:
            If ``preds`` or ``target`` contain non-string values.

    Example::
        Basic example with two strings. Going from “rain” -> “sain” -> “shin” -> “shine” takes 3 edits:

        >>> from torchmetrics.functional.text import edit_distance
        >>> edit_distance(["rain"], ["shine"])
        tensor(3.)

    Example::
        Basic example with two strings and substitution cost of 2. Going from “rain” -> “sain” -> “shin” -> “shine”
        takes 3 edits, where two of them are substitutions:

        >>> from torchmetrics.functional.text import edit_distance
        >>> edit_distance(["rain"], ["shine"], substitution_cost=2)
        tensor(5.)

    Example::
        Multiple strings example:

        >>> from torchmetrics.functional.text import edit_distance
        >>> edit_distance(["rain", "lnaguaeg"], ["shine", "language"], reduction=None)
        tensor([3, 4], dtype=torch.int32)
        >>> edit_distance(["rain", "lnaguaeg"], ["shine", "language"], reduction="mean")
        tensor(3.5000)

    """
    distance = _edit_distance_update(preds, target, substitution_cost)
    return _edit_distance_compute(distance, num_elements=distance.numel(), reduction=reduction)
