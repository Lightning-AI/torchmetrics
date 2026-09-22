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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def _tie_average_dcg(target: Tensor, preds: Tensor, discount_cumsum: Tensor) -> Tensor:
    """Translated version of sklearns `_tie_average_dcg` function.

    Args:
        target: ground truth about each document relevance.
        preds: estimated probabilities of each document to be relevant.
        discount_cumsum: cumulative sum of the discount.

    Returns:
        The cumulative gain of the tied elements.

    """
    _, inv, counts = torch.unique(-preds, return_inverse=True, return_counts=True)
    ranked = torch.zeros_like(counts, dtype=torch.float32)
    ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))
    ranked = ranked / counts
    groups = counts.cumsum(dim=0) - 1
    discount_sums = torch.zeros_like(counts, dtype=torch.float32)
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = discount_cumsum[groups].diff()
    return (ranked * discount_sums).sum()


def _dcg_sample_scores(target: Tensor, preds: Tensor, top_k: int, ignore_ties: bool) -> Tensor:
    """Translated version of sklearns `_dcg_sample_scores` function.

    Args:
        target: ground truth about each document relevance.
        preds: estimated probabilities of each document to be relevant.
        top_k: consider only the top k elements
        ignore_ties: If True, ties are ignored. If False, ties are averaged.

    Returns:
        The cumulative gain

    """
    discount = 1.0 / (torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0))
    discount[top_k:] = 0.0

    if ignore_ties:
        ranking = preds.argsort(descending=True)
        ranked = target[ranking]
        cumulative_gain = (discount * ranked).sum()
    else:
        discount_cumsum = discount.cumsum(dim=-1)
        cumulative_gain = _tie_average_dcg(target, preds, discount_cumsum)
    return cumulative_gain


def _handle_empty_target(action: str, device: torch.device) -> Optional[Tensor]:
    """Return a default nDCG score when the target contains no positive labels.

    Args:
        action: policy for handling empty targets:
            - "skip": return None (exclude from batch average)
            - "pos": return a score of 1.0
            - "neg": return a score of 0.0
        device: the torch device on which to create the output tensor.

    Returns:
        A scalar tensor with the default score if action is "pos" or "neg".
        None if action is "skip".

    Raises:
        ValueError: if ``action`` is not one of {"skip", "pos", "neg"}.

    """
    if action == "skip":
        return None
    if action == "pos":
        return torch.tensor(1.0, device=device)
    if action == "neg":
        return torch.tensor(0.0, device=device)
    raise ValueError(f"Invalid empty_target_action: {action}")


def retrieval_normalized_dcg(
    preds: Tensor,
    target: Tensor,
    top_k: Optional[int] = None,
    empty_target_action: str = "skip",
) -> Tensor:
    """Compute `Normalized Discounted Cumulative Gain`_ (for information retrieval).

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        top_k: consider only the top k elements (default: ``None``, which considers them all)
        empty_target_action: what to do when the target has no positives:
            - "skip": exclude from average
            - "pos": assign score 1.0
            - "neg": assign score 0.0

    Return:
        A single-value tensor with the nDCG of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_normalized_dcg
        >>> preds = torch.tensor([.1, .2, .3, 4, 70])
        >>> target = torch.tensor([10, 0, 0, 1, 5])
        >>> retrieval_normalized_dcg(preds, target)
        tensor(0.6957)

    """
    original_shape = preds.shape
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    # reshape back if input was 2D
    if len(original_shape) == 2:
        preds = preds.view(original_shape)
        target = target.view(original_shape)
    else:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

    n_samples, n_labels = preds.shape
    top_k = n_labels if top_k is None else top_k
    top_k = min(top_k, n_labels)

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    scores = []
    for p, t in zip(preds, target):
        gain = _dcg_sample_scores(t, p, top_k, ignore_ties=False)
        ideal_gain = _dcg_sample_scores(t, t, top_k, ignore_ties=True)

        if ideal_gain == 0:
            score = _handle_empty_target(empty_target_action, preds.device)
            if score is not None:
                scores.append(score)
        else:
            scores.append(gain / ideal_gain)

    if not scores:
        return torch.tensor(0.0, device=preds.device)

    return torch.stack(scores).mean()
