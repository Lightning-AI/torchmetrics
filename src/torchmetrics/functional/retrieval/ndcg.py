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


def _tie_average_dcg(target: Tensor, preds: Tensor, discount: Tensor) -> Tensor:
    """Compute DCG for tied predictions using scatter operations.

    Replaces the ``torch.unique`` approach with ``diff`` + ``scatter_add_``, which is
    significantly faster on GPU (``torch.unique`` is ~15x slower on GPU than CPU).

    Args:
        target: ground truth relevances, shape ``(L,)`` or ``(B, L)``.
        preds: predicted scores, shape ``(L,)`` or ``(B, L)``.
        discount: per-rank discount values ``1 / log2(rank + 2)``, shape ``(L,)``.

    Returns:
        DCG value(s): scalar for 1-D input, shape ``(B,)`` for batched input.

    """
    batched = preds.dim() > 1
    B = preds.shape[0] if batched else 1
    L = preds.shape[-1]

    if not batched:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

    # Sort each row by descending predicted score
    order = preds.argsort(dim=-1, descending=True, stable=True)
    p_sorted = preds.gather(-1, order)
    g_sorted = target.float().gather(-1, order)

    # Detect tie-group boundaries: True at the first element of each new group
    new_grp = torch.cat(
        [
            torch.ones(B, 1, dtype=torch.bool, device=preds.device),
            p_sorted.diff(dim=-1) != 0,
        ],
        dim=-1,
    )  # (B, L)

    # Per-element group id, made unique across the batch
    gid = new_grp.long().cumsum(-1) - 1  # 0-based within each row
    gid = gid + torch.arange(B, device=preds.device).unsqueeze(-1) * L

    # Scatter: accumulate gains, discounts, and counts per group
    flat_gid = gid.flatten()
    flat_gain = g_sorted.flatten().float()
    flat_disc = discount.unsqueeze(0).expand(B, -1).flatten().float()

    grp_gain = torch.zeros(B * L, dtype=torch.float32, device=preds.device)
    grp_disc = torch.zeros(B * L, dtype=torch.float32, device=preds.device)
    grp_cnt = torch.zeros(B * L, dtype=torch.long, device=preds.device)

    grp_gain.scatter_add_(0, flat_gid, flat_gain)
    grp_disc.scatter_add_(0, flat_gid, flat_disc)
    grp_cnt.scatter_add_(0, flat_gid, torch.ones_like(flat_gid))

    contrib = grp_gain * grp_disc / grp_cnt.float().clamp(min=1)
    dcg = contrib.view(B, L).sum(-1)  # (B,)
    return dcg if batched else dcg.squeeze(0)


def _dcg_sample_scores(target: Tensor, preds: Tensor, top_k: int, ignore_ties: bool) -> Tensor:
    """Compute DCG sample scores.

    Args:
        target: ground truth relevances, shape ``(L,)`` or ``(B, L)``.
        preds: predicted scores, shape ``(L,)`` or ``(B, L)``.
        top_k: consider only the top k elements.
        ignore_ties: If ``True``, ties are broken by order. If ``False``, ties are averaged.

    Returns:
        DCG value(s): scalar for 1-D input, shape ``(B,)`` for batched input.

    """
    L = target.shape[-1]
    discount = 1.0 / torch.log2(torch.arange(L, device=target.device) + 2.0)
    discount[top_k:] = 0.0

    if ignore_ties:
        ranking = preds.argsort(dim=-1, descending=True)
        ranked = target.float().gather(-1, ranking)
        return (discount * ranked).sum(-1)

    return _tie_average_dcg(target, preds, discount)


def retrieval_normalized_dcg(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Compute `Normalized Discounted Cumulative Gain`_ (for information retrieval).

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

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
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    top_k = preds.shape[-1] if top_k is None else top_k

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    gain = _dcg_sample_scores(target, preds, top_k, ignore_ties=False)
    normalized_gain = _dcg_sample_scores(target, target, top_k, ignore_ties=True)

    # filter undefined scores
    all_irrelevant = normalized_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalized_gain[~all_irrelevant]

    return gain.mean()
