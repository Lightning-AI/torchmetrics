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
from typing import Optional, Tuple

import torch
from torch import Tensor


def _rank_data(x: Tensor) -> Tensor:
    """Rank data based on values."""
    # torch.unique does not support input that requires grad
    with torch.no_grad():
        _, inverse, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    ranks = counts.cumsum(dim=0)
    return ranks[inverse]


def _check_ranking_input(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
    """Check that ranking input have the correct dimensions."""
    if preds.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Expected both predictions and target to matrices of shape `[N,C]`"
            f" but got {preds.ndim} and {target.ndim}"
        )
    if preds.shape != target.shape:
        raise ValueError("Expected both predictions and target to have same shape")
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != preds.shape[0]:
            raise ValueError(
                "Expected sample weights to be 1 dimensional and have same size"
                f" as the first dimension of preds and target but got {sample_weight.shape}"
            )


def _coverage_error_update(
    preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None
) -> Tuple[Tensor, int, Optional[Tensor]]:
    """Accumulate state for coverage error
    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels
        sample_weight: optional tensor with weight for each sample

    """
    _check_ranking_input(preds, target, sample_weight)
    offset = torch.zeros_like(preds)
    offset[target == 0] = preds.min().abs() + 10  # Any number >1 works
    preds_mod = preds + offset
    preds_min = preds_mod.min(dim=1)[0]
    coverage = (preds >= preds_min[:, None]).sum(dim=1).to(torch.float32)
    if isinstance(sample_weight, Tensor):
        coverage *= sample_weight
        sample_weight = sample_weight.sum()
    return coverage.sum(), coverage.numel(), sample_weight


def _coverage_error_compute(coverage: Tensor, n_elements: int, sample_weight: Optional[Tensor] = None) -> Tensor:
    if sample_weight is not None and sample_weight != 0.0:
        return coverage / sample_weight
    return coverage / n_elements


def coverage_error(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
    """Computes multilabel coverage error [1]. The score measure how far we need to go through the ranked scores to
    cover all true labels. The best value is equal to the average number of labels in the target tensor per sample.

    Args:
        preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should either be probabilities of the positive class or corresponding logits
        target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should only contain binary labels.
        sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
            should be weighted in the final score.

    Example:
        >>> from torchmetrics.functional import coverage_error
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> coverage_error(preds, target)
        tensor(3.9000)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """
    coverage, n_elements, sample_weight = _coverage_error_update(preds, target, sample_weight)
    return _coverage_error_compute(coverage, n_elements, sample_weight)


def _label_ranking_average_precision_update(
    preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None
) -> Tuple[Tensor, int, Optional[Tensor]]:
    """Accumulate state for label ranking average precision.

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels
        sample_weight: optional tensor with weight for each sample
    """
    _check_ranking_input(preds, target, sample_weight)
    # Invert so that the highest score receives rank 1
    neg_preds = -preds

    score = torch.tensor(0.0, device=neg_preds.device)
    n_preds, n_labels = neg_preds.shape
    for i in range(n_preds):
        relevant = target[i] == 1
        ranking = _rank_data(neg_preds[i][relevant]).float()
        if len(ranking) > 0 and len(ranking) < n_labels:
            rank = _rank_data(neg_preds[i])[relevant].float()
            score_idx = (ranking / rank).mean()
        else:
            score_idx = 1.0

        if sample_weight is not None:
            score_idx *= sample_weight[i]

        score += score_idx

    return score, n_preds, sample_weight.sum() if isinstance(sample_weight, Tensor) else sample_weight


def _label_ranking_average_precision_compute(
    score: Tensor, n_elements: int, sample_weight: Optional[Tensor] = None
) -> Tensor:
    """Computes the final label ranking average precision score."""
    if sample_weight is not None and sample_weight != 0.0:
        return score / sample_weight
    return score / n_elements


def label_ranking_average_precision(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
    """Computes label ranking average precision score for multilabel data [1]. The score is the average over each
    ground truth label assigned to each sample of the ratio of true vs. total labels with lower score. Best score
    is 1.

    Args:
        preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should either be probabilities of the positive class or corresponding logits
        target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should only contain binary labels.
        sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
            should be weighted in the final score.

    Example:
        >>> from torchmetrics.functional import label_ranking_average_precision
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> label_ranking_average_precision(preds, target)
        tensor(0.7744)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """
    score, n_elements, sample_weight = _label_ranking_average_precision_update(preds, target, sample_weight)
    return _label_ranking_average_precision_compute(score, n_elements, sample_weight)


def _label_ranking_loss_update(
    preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None
) -> Tuple[Tensor, int, Optional[Tensor]]:
    """Accumulate state for label ranking loss.

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels
        sample_weight: optional tensor with weight for each sample
    """
    _check_ranking_input(preds, target, sample_weight)
    n_preds, n_labels = preds.shape
    relevant = target == 1
    n_relevant = relevant.sum(dim=1)

    # Ignore instances where number of true labels is 0 or n_labels
    mask = (n_relevant > 0) & (n_relevant < n_labels)
    preds = preds[mask]
    relevant = relevant[mask]
    n_relevant = n_relevant[mask]

    # Nothing is relevant
    if len(preds) == 0:
        return torch.tensor(0.0, device=preds.device), 1, sample_weight

    inverse = preds.argsort(dim=1).argsort(dim=1)
    per_label_loss = ((n_labels - inverse) * relevant).to(torch.float32)
    correction = 0.5 * n_relevant * (n_relevant + 1)
    denom = n_relevant * (n_labels - n_relevant)
    loss = (per_label_loss.sum(dim=1) - correction) / denom
    if isinstance(sample_weight, Tensor):
        loss *= sample_weight[mask]
        sample_weight = sample_weight.sum()
    return loss.sum(), n_preds, sample_weight


def _label_ranking_loss_compute(loss: Tensor, n_elements: int, sample_weight: Optional[Tensor] = None) -> Tensor:
    """Computes the final label ranking loss."""
    if sample_weight is not None and sample_weight != 0.0:
        return loss / sample_weight
    return loss / n_elements


def label_ranking_loss(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
    """Computes the label ranking loss for multilabel data [1]. The score is corresponds to the average number of
    label pairs that are incorrectly ordered given some predictions weighted by the size of the label set and the
    number of labels not in the label set. The best score is 0.

    Args:
        preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should either be probabilities of the positive class or corresponding logits
        target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
            of labels. Should only contain binary labels.
        sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
            should be weighted in the final score.

    Example:
        >>> from torchmetrics.functional import label_ranking_loss
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> label_ranking_loss(preds, target)
        tensor(0.4167)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """
    loss, n_element, sample_weight = _label_ranking_loss_update(preds, target, sample_weight)
    return _label_ranking_loss_compute(loss, n_element, sample_weight)
