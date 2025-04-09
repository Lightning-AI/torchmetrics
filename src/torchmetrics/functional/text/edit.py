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
    preds: Union[str, Sequence[str], Tensor, Sequence[Tensor]],
    target: Union[str, Sequence[str], Tensor, Sequence[Tensor]],
    substitution_cost: int = 1,
) -> Tensor:
    """Update the edit distance score with the current set of predictions and targets.

    Args:
        preds: An iterable of predicted texts (strings) or a tensor of categorical values or a list of tensors
        target: An iterable of reference texts (strings) or a tensor of categorical values or a list of tensors
        substitution_cost: The cost of substituting one character for another.

    Returns:
        A tensor containing the edit distance scores for each prediction-target pair.

    """
    # Handle tensor inputs
    if isinstance(preds, Tensor) and isinstance(target, Tensor):
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if preds.size(0) != target.size(0):
            raise ValueError(
                f"Expected argument `preds` and `target` to have same batch size, but got {preds.size(0)} and {target.size(0)}"
            )
        # Convert tensors to lists of lists for the edit distance algorithm
        preds = [p.tolist() for p in preds]
        target = [t.tolist() for t in target]
    # Handle lists of tensors
    elif isinstance(preds, (list, tuple)) and isinstance(target, (list, tuple)):
        if not all(isinstance(x, Tensor) for x in preds):
            raise ValueError(f"Expected all values in argument `preds` to be tensor type, but got {preds}")
        if not all(isinstance(x, Tensor) for x in target):
            raise ValueError(f"Expected all values in argument `target` to be tensor type, but got {target}")
        if len(preds) != len(target):
            raise ValueError(
                f"Expected argument `preds` and `target` to have same length, but got {len(preds)} and {len(target)}"
            )
        # Convert tensors to lists for the edit distance algorithm
        preds = [p.tolist() for p in preds]
        target = [t.tolist() for t in target]
    else:
        # Handle string inputs (existing behavior)
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
    preds: Union[str, Sequence[str], Tensor],
    target: Union[str, Sequence[str], Tensor],
    substitution_cost: int = 1,
    reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
) -> Tensor:
    """Calculates the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to transform the
    predicted text into the reference text. The lower the distance, the more accurate the model is considered to be.

    Implementation is similar to `nltk.edit_distance <https://www.nltk.org/_modules/nltk/metrics/distance.html>`_.

    Args:
        preds: An iterable of predicted texts (strings) or a tensor of categorical values
        target: An iterable of reference texts (strings) or a tensor of categorical values
        substitution_cost: The cost of substituting one character for another.
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Raises:
        ValueError:
            If ``preds`` and ``target`` do not have the same length/batch size.
        ValueError:
            If ``preds`` or ``target`` contain non-string values when using string inputs.

    Example::
        Basic example with two strings. Going from "rain" -> "sain" -> "shin" -> "shine" takes 3 edits:

        >>> from torchmetrics.functional.text import edit_distance
        >>> edit_distance(["rain"], ["shine"])
        tensor(3.)

    Example::
        Basic example with two strings and substitution cost of 2. Going from "rain" -> "sain" -> "shin" -> "shine"
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

    Example::
        Using tensors of categorical values:

        >>> from torchmetrics.functional.text import edit_distance
        >>> preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> target = torch.tensor([[1, 2, 4], [4, 5, 7]])
        >>> edit_distance(preds, target)
        tensor(2.0000)

    """
    distance = _edit_distance_update(preds, target, substitution_cost)
    return _edit_distance_compute(distance, num_elements=distance.numel(), reduction=reduction)
