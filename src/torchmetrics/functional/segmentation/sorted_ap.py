# Copyright The PyTorch Lightning team.
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
#
# The sortedAP metric from the paper  "sortedAP: Rethinking evaluation
# metrics for instance segmentation", Long Chen et al. is implemented
# based on the reference implementation by Long Chen:
# https://github.com/looooongchen/sortedap

import warnings

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE

# threshold used to build the maximal (loosest) matching between predictions and targets
_MATCH_THRESHOLD = 1e-6


def _sorted_ap_validate_args(match_method: str) -> None:
    """Validate the input arguments of the ``sorted_ap`` metric."""
    if match_method not in ("hungarian", "greedy"):
        raise ValueError(
            f"Expected argument `match_method` to be one of 'hungarian', 'greedy', but got {match_method}."
        )


def _sorted_ap_check_input(preds: list[Tensor], target: list[Tensor]) -> None:
    """Check that ``preds`` and ``target`` are lists of equal length with valid mask tensors."""
    if not isinstance(preds, (list, tuple)) or not isinstance(target, (list, tuple)):
        raise ValueError(
            "Expected both `preds` and `target` to be a list of tensors, one entry per image, "
            f"but got {type(preds)} and {type(target)}."
        )
    if len(preds) != len(target):
        raise ValueError(
            f"Expected `preds` and `target` to have the same length (number of images), "
            f"but got {len(preds)} and {len(target)}."
        )
    for p, t in zip(preds, target):
        if p.ndim != 3 or t.ndim != 3:
            raise ValueError(
                "Expected each image in `preds` and `target` to be a 3D tensor of stacked binary masks "
                f"with shape `(num_instances, height, width)`, but got shapes {tuple(p.shape)} and {tuple(t.shape)}."
            )
        if p.shape[1:] != t.shape[1:]:
            raise ValueError(
                "Expected the spatial dimensions of `preds` and `target` to match, "
                f"but got {tuple(p.shape[1:])} and {tuple(t.shape[1:])}."
            )


def _jaccard_matrix(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the pairwise Jaccard (IoU) matrix between predicted and target instance masks.

    Args:
        preds: Boolean tensor of shape ``(num_preds, height, width)``.
        target: Boolean tensor of shape ``(num_targets, height, width)``.

    Returns:
        A tensor of shape ``(num_preds, num_targets)`` with the Jaccard index for each pair.

    """
    preds = preds.flatten(1).float()  # (num_preds, H*W)
    target = target.flatten(1).float()  # (num_targets, H*W)
    intersection = preds @ target.T  # (num_preds, num_targets)
    area_preds = preds.sum(dim=1, keepdim=True)  # (num_preds, 1)
    area_target = target.sum(dim=1, keepdim=True).T  # (1, num_targets)
    union = area_preds + area_target - intersection
    return _safe_divide(intersection, union, zero_division="nan")


def _match_hungarian(jaccard: Tensor) -> Tensor:
    """Return a boolean match matrix using Hungarian (linear sum) assignment on the Jaccard matrix."""
    from scipy.optimize import linear_sum_assignment

    cost = jaccard.clone()
    cost[cost < _MATCH_THRESHOLD] = 0.0
    match = torch.zeros_like(jaccard, dtype=torch.bool)

    # restrict the problem to rows/columns that have at least one eligible overlap
    idx_pd = cost.amax(dim=1) > 0
    idx_gt = cost.amax(dim=0) > 0
    if not bool(idx_pd.any()) or not bool(idx_gt.any()):
        return match

    sub_cost = cost[idx_pd][:, idx_gt]
    row, col = linear_sum_assignment((1 - sub_cost).cpu().numpy())
    row = torch.nonzero(idx_pd, as_tuple=False).flatten()[row]
    col = torch.nonzero(idx_gt, as_tuple=False).flatten()[col]

    # keep only assignments that actually overlap (guard against zero-overlap pairings)
    keep = jaccard[row, col] >= _MATCH_THRESHOLD
    match[row[keep], col[keep]] = True
    return match


def _match_greedy(jaccard: Tensor) -> Tensor:
    """Return a boolean match matrix using a greedy one-to-one assignment on the Jaccard matrix."""
    match = torch.zeros_like(jaccard, dtype=torch.bool)
    work = jaccard.clone()
    work[work < _MATCH_THRESHOLD] = 0.0
    used_preds = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    used_target = torch.zeros(work.shape[1], dtype=torch.bool, device=work.device)

    while True:
        flat_idx = int(torch.argmax(work))
        best = work.flatten()[flat_idx]
        if best <= 0:
            break
        r = flat_idx // work.shape[1]
        c = flat_idx % work.shape[1]
        match[r, c] = True
        used_preds[r] = True
        used_target[c] = True
        work[r, :] = 0.0
        work[:, c] = 0.0
    return match


def _sorted_ap_update(
    preds: list[Tensor],
    target: list[Tensor],
    match_method: Literal["hungarian", "greedy"] = "hungarian",
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the per-batch state contributions for the ``sorted_ap`` metric.

    For every image, the maximal matching between predictions and targets is computed at a very loose
    threshold, and the Jaccard values of the matched pairs are collected. These, together with the number
    of true positives, false negatives and predictions, are accumulated across the whole dataset.

    Args:
        preds: List of predicted instance masks, one tensor of shape ``(num_preds, height, width)`` per image.
        target: List of target instance masks, one tensor of shape ``(num_targets, height, width)`` per image.
        match_method: The matching method used to assign predictions to targets.

    Returns:
        A 3-tuple ``(jaccard, tp0, fn0)`` where ``jaccard`` is a 1D tensor of matched Jaccard values pooled
        over the batch, ``tp0`` is the number of matched pairs (true positives at the loose threshold) plus
        the total number of predictions encoded together for accumulation.

    """
    use_hungarian = match_method == "hungarian"
    if use_hungarian and not _SCIPY_AVAILABLE:
        warnings.warn(
            "Argument `match_method='hungarian'` requires `scipy` to be installed. "
            "Falling back to `match_method='greedy'`.",
            UserWarning,
            stacklevel=2,
        )
        use_hungarian = False

    matched_jaccards: list[Tensor] = []
    tp0 = 0
    fn0 = 0
    n_pd = 0

    for p, t in zip(preds, target):
        p = p.bool()
        t = t.bool()

        # drop empty instance channels (masks with no positive pixels)
        p = p[p.flatten(1).any(dim=1)]
        t = t[t.flatten(1).any(dim=1)]

        num_pd, num_gt = p.shape[0], t.shape[0]
        n_pd += num_pd

        if num_pd == 0 or num_gt == 0:
            fn0 += num_gt
            continue

        jaccard = _jaccard_matrix(p, t)
        match = _match_hungarian(jaccard) if use_hungarian else _match_greedy(jaccard)

        n_match = int(match.sum())
        tp0 += n_match
        fn0 += num_gt - n_match
        if n_match > 0:
            matched_jaccards.append(jaccard[match])

    jaccard = torch.cat(matched_jaccards) if matched_jaccards else torch.empty(0)
    counts = torch.tensor([tp0, fn0, n_pd], dtype=torch.long, device=jaccard.device)
    return jaccard, counts, torch.tensor(len(preds), device=jaccard.device)


def _sorted_ap_compute(jaccard: Tensor, counts: Tensor) -> Tensor:
    """Compute the final sortedAP score from accumulated state, following Algorithm 2 of the paper.

    The score is the area under the (IoU-threshold vs. average-precision) curve, computed with the
    trapezoidal rule. A left boundary rectangle from ``x=0`` to the smallest matched IoU is added at the
    initial precision ``AP_0 = TP0 / (P + FN0)``. No right boundary term is needed because the final
    precision ``AP_TP0`` is zero.

    Args:
        jaccard: 1D tensor of all matched IoU (Jaccard) values pooled over the dataset.
        counts: 1D long tensor ``[tp0, fn0, n_pd]`` accumulated over the dataset, where ``n_pd`` is the
            total number of predictions ``P``.

    Returns:
        The scalar sortedAP score in the range ``[0, 1]``.

    """
    tp0 = int(counts[0])
    fn0 = int(counts[1])
    n_pd = int(counts[2])

    # degenerate cases
    if tp0 == 0 and fn0 != 0:
        return torch.tensor(0.0)
    if n_pd == 0 and fn0 == 0:
        return torch.tensor(1.0)
    if jaccard.numel() == 0:
        return torch.tensor(0.0)

    ious, _ = torch.sort(jaccard)  # [IoU_1, ..., IoU_TP0] ascending
    device = ious.device

    # AP_0 = TP0 / (P + FN0); left boundary rectangle from x=0 to x=IoU_1 at height AP_0
    ap_0 = tp0 / (n_pd + fn0)
    score = ious[0] * ap_0

    # k = 1..TP0 : AP_k = (TP0 - k) / (P + FN0 + k), paired with t_k = IoU_k (AP_TP0 == 0)
    k = torch.arange(1, tp0 + 1, dtype=torch.float, device=device)
    ap = (tp0 - k) / (n_pd + fn0 + k)  # AP_1 .. AP_TP0

    # previous points for each step: (AP_0, IoU_1) then (AP_{k-1}, IoU_{k-1})
    ap_prev = torch.cat([torch.tensor([ap_0], device=device), ap[:-1]])
    t_prev = torch.cat([ious[:1], ious[:-1]])

    # sum of trapezoids: 0.5 * (t_k - t_{k-1}) * (AP_k + AP_{k-1})
    score = score + 0.5 * torch.sum((ious - t_prev) * (ap + ap_prev))
    return score.float()


def sorted_ap(
    preds: list[Tensor],
    target: list[Tensor],
    match_method: Literal["hungarian", "greedy"] = "hungarian",
) -> Tensor:
    """Compute the sortedAP metric for instance segmentation.

    The sortedAP metric is a threshold-free summary of average precision for instance segmentation. Instead of
    reporting AP at a fixed Jaccard (IoU) threshold, predictions and targets are matched at a very loose
    threshold, and the average precision is integrated over all possible thresholds by sweeping the Jaccard
    values of the matched pairs. The final score is the area under the resulting threshold-vs-precision curve.

    Args:
        preds: List of predicted instance masks, one tensor of stacked binary masks of shape
            ``(num_preds, height, width)`` per image.
        target: List of target instance masks, one tensor of stacked binary masks of shape
            ``(num_targets, height, width)`` per image.
        match_method: The matching method used to assign predictions to targets. Can be one of
            ``"hungarian"`` (default, requires ``scipy``) or ``"greedy"``. If ``"hungarian"`` is
            requested but ``scipy`` is not installed, the metric falls back to ``"greedy"``.

    Returns:
        The scalar sortedAP score in the range ``[0, 1]``.

    Example::
        >>> import torch
        >>> from torchmetrics.functional.segmentation import sorted_ap
        >>> preds = [torch.tensor([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        ...                        [[0, 0, 0], [0, 1, 1], [0, 0, 0]]], dtype=torch.bool)]
        >>> target = [torch.tensor([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        ...                         [[0, 0, 0], [0, 1, 1], [0, 0, 0]]], dtype=torch.bool)]
        >>> sorted_ap(preds, target)
        tensor(1.)

    """
    _sorted_ap_validate_args(match_method)
    _sorted_ap_check_input(preds, target)
    jaccard, counts, _ = _sorted_ap_update(preds, target, match_method)
    return _sorted_ap_compute(jaccard, counts)
