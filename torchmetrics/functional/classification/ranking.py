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
from torchmetrics.utilities.checks import _input_format_classification


def rank_data(x: Tensor) -> Tensor:
    _, inverse, counts = torch.unique(
        x, sorted=True, return_inverse=True, return_counts=True)
    ranks = counts.cumsum(dim=0)
    return ranks[inverse]


def _check_ranking_input(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor


    return preds

def _coverage_error_update(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tuple[Tensor, int, Optional[Tensor]]:
    preds, target = _input_format_classification(
        preds: Tensor,
        target: Tensor,
        threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,

    )
    offset = torch.zeros_like(preds)
    offset[target == 0] = 1.1  # Any number >1 works
    preds_mod = preds + offset
    preds_min = preds_mod.min(dim=1)[0]
    coverage = (preds >= preds_min[:, None]).sum(dim=1).to(torch.float32)
    if isinstance(sample_weight, Tensor):
        coverage *= sample_weight
        sample_weight = sample_weight.sum()
    return coverage.sum(), coverage.numel(), sample_weight


def _coverage_error_compute(coverage: Tensor, n_elements: int, sample_weight: Optional[Tensor] = None) -> Tensor:
    if sample_weight is not None:
        return coverage / sample_weight
    return coverage / n_elements


def coverage_error(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
    coverage, n_elements, sample_weight = _coverage_error_update(preds, target, sample_weight)
    return _coverage_error_compute(coverage, n_elements, sample_weight)


def coverage_error(y_pred, y_true, sample_weights=None):
    offset = torch.zeros_like(y_pred)
    offset[y_true == 0] = 1.1  # Any number >1 works
    y_pred_mod = y_pred + offset
    y_pred_min = y_pred_mod.min(dim=1)[0]
    coverage = (y_pred >= y_pred_min[:, None]).sum(dim=1).to(torch.float32)

    if sample_weights is not None:
        coverage *= sample_weights
        return coverage.sum() / sample_weights.sum()

    return coverage.mean()


def _label_ranking_average_precision_update(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None):
    # Invert so that the highest score receives rank 1
    preds = -preds
    relevant = 

def _label_ranking_average_precision_compute():


def label_ranking_average_precision(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:



def label_ranking_average_precision(y_pred, y_true, sample_weights=None):
    # Invert so that the highest score receives rank 1
    y_pred = -y_pred

    score = torch.tensor(0.0, device=y_pred.device)
    n_preds, n_labels = y_pred.shape
    for i in range(n_preds):
        relevant = y_true[i] == 1
        L = rank_data(y_pred[i][relevant])
        if len(L) > 0 and len(L) < n_labels:
            rank = rank_data(y_pred[i])[relevant]
            score_i = (L / rank).mean()
        else:
            score_i = 1.0

        if sample_weights is not None:
            score_i *= sample_weights[i]

        score += score_i

    if sample_weights is None:
        score /= n_preds
    else:
        score /= sample_weights.sum()
    return score



def _label_ranking_loss_update(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None):
    n_labels = preds.shape[1]
    relevant = target == 1
    n_relevant = relevant.sum(dim=1)

    # Ignore instances where number of true labels is 0 or n_labels
    mask = (n_relevant > 0) & (n_relevant < n_labels)
    preds = preds[mask]
    relevant = relevant[mask]
    n_relevant = n_relevant[mask]

    # Nothing is relevant
    if len(preds) == 0:
        return torch.tensor(0.0, device=preds.device)

    inverse = preds.argsort(dim=1).argsort(dim=1)
    per_label_loss = ((n_labels - inverse) * relevant).to(torch.float32)
    correction = 0.5 * n_relevant * (n_relevant + 1)
    denom = n_relevant * (n_labels - n_relevant)
    loss = (per_label_loss.sum(dim=1) - correction) / denom

    if sample_weights is not None:
        coverage *= sample_weights
        return coverage.sum() / sample_weights.sum()


def _label_ranking_loss_compute():


def label_ranking_loss(preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:



def label_ranking_loss(y_pred, y_true, sample_weights=None):
    n_labels = y_pred.shape[1]
    relevant = y_true == 1
    n_relevant = relevant.sum(dim=1)

    # Ignore instances where number of true labels is 0 or n_labels
    mask = (n_relevant > 0) & (n_relevant < n_labels)
    y_pred = y_pred[mask]
    relevant = relevant[mask]
    n_relevant = n_relevant[mask]

    if len(y_pred) == 0:
        return torch.tensor(0.0, device=y_pred.device)

    inverse = y_pred.argsort(dim=1).argsort(dim=1)
    per_label_loss = ((n_labels - inverse) * relevant).to(torch.float32)
    correction = 0.5 * n_relevant * (n_relevant + 1)  # Sum of 1..n
    denom = n_relevant * (n_labels - n_relevant)
    loss = (per_label_loss.sum(dim=1) - correction) / denom

    if sample_weights is not None:
        loss *= sample_weights[mask]
        return loss.sum() / sample_weights.sum()

    return loss.mean()


