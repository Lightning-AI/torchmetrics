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

from typing import Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.compute import normalize_logits_if_needed


def _brier_decomposition(
    probabilities: torch.Tensor = None,
    confusion_matrix: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose the Brier score into uncertainty, resolution, and reliability.

    [Proper scoring rules][1] measure the quality of probabilistic predictions;
    any proper scoring rule admits a [unique decomposition][2] as
    `Score = Uncertainty - Resolution + Reliability`, where:

    * `Uncertainty`, is a generalized entropy of the average predictive
      distribution; it can both be positive or negative.
    * `Resolution`, is a generalized variance of individual predictive
      distributions; it is always non-negative. Difference in predictions reveal
      information, that is why a larger resolution improves the predictive score.
    * `Reliability`, a measure of calibration of predictions against the true
      frequency of events. It is always non-negative and a lower value here
      indicates better calibration.

    Args:
      labels: Tensor, (n,), with torch.int64 elements containing ground
        truth class labels in the range [0, nlabels].
      logits: Tensor, (n, nlabels), with logits for n instances and nlabels.
      probabilities: Tensor, (n, nlabels), with predictive probability
        distribution (alternative to logits argument).
      confusion_matrix: Tensor, (nlabels, nlabels), the confusion matrix.

    Returns:
      uncertainty: Tensor, scalar, the uncertainty component of the
        decomposition.
      resolution: Tensor, scalar, the resolution component of the decomposition.
      reliability: Tensor, scalar, the reliability component of the
        decomposition.

    """
    n, nlabels = probabilities.shape  # Implicit rank check.

    confusion_matrix = confusion_matrix.T

    # Compute pbar, the average distribution
    pred_class = torch.argmax(probabilities, dim=1)
    dist_weights = confusion_matrix.sum(dim=1)
    dist_weights /= dist_weights.sum()
    pbar = confusion_matrix.sum(dim=0)
    pbar /= pbar.sum()

    # dist_mean[k,:] contains the empirical distribution for the set M_k
    # Some outcomes may not realize, corresponding to dist_weights[k] = 0
    dist_mean = confusion_matrix / (confusion_matrix.sum(dim=1, keepdim=True) + 1.0e-7)

    # Uncertainty: quadratic entropy of the average label distribution
    uncertainty = torch.sum(pbar - pbar**2)

    # Resolution: expected quadratic divergence of predictive to mean
    resolution = (pbar.unsqueeze(1) - dist_mean) ** 2
    resolution = torch.sum(dist_weights * resolution.sum(dim=1))

    # Reliability: expected quadratic divergence of predictive to true
    prob_true = dist_mean[pred_class]
    reliability = torch.sum((prob_true - probabilities) ** 2, dim=1)
    reliability = torch.mean(reliability)

    return uncertainty, resolution, reliability


def _mean_brier_score(labels: torch.Tensor, probabilities: torch.Tensor = None) -> torch.Tensor:
    """Compute elementwise Brier score.

    The Brier score is a proper scoring rule that measures the accuracy of probabilistic predictions.
    It is calculated as the squared difference between the predicted probability distribution and
    the actual outcome.

    Args:
        labels (torch.Tensor): Tensor of integer labels with shape [N1, N2, ...].
        probs (torch.Tensor, optional): Tensor of categorical probabilities with shape [N1, N2, ..., M].
        logits (torch.Tensor, optional): If `probs` is None, class probabilities are computed as a
            softmax over these logits. This argument is ignored if `probs` is provided.

    Returns:
        torch.Tensor: Tensor of shape [N1, N2, ...] consisting of the Brier score contribution
        from each element. The full-dataset Brier score is the average of these values.

    """
    nlabels = probabilities.shape[-1]
    flat_probabilities = probabilities.view(-1, nlabels)
    flat_labels = labels.view(-1)

    # Gather the probabilities corresponding to the true labels
    plabel = flat_probabilities[torch.arange(len(flat_labels)), flat_labels]
    out = torch.sum(flat_probabilities**2, dim=-1) - 2 * plabel + 1

    return out.view(labels.shape).mean()


def _mean_brier_score_and_decomposition(
    labels: torch.Tensor,
    probabilities: torch.Tensor = None,
    confusion_matrix: torch.Tensor = None,
) -> dict[str, torch.Tensor]:
    mean_brier = _mean_brier_score(labels, probabilities)
    uncertainty, resolution, reliability = _brier_decomposition(probabilities, confusion_matrix)

    return {
        "MeanBrier": mean_brier,
        "Uncertainty": uncertainty,
        "Resolution": resolution,
        "Reliability": reliability,
    }


def _adjust_threshold_arg(
    thresholds: Optional[Union[int, list[float], Tensor]] = None, device: Optional[torch.device] = None
) -> Optional[Tensor]:
    """Convert threshold arg for list and int to tensor format."""
    if isinstance(thresholds, int):
        return torch.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        return torch.tensor(thresholds, device=device)
    return thresholds


def _binary_brier_format(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
    normalization: Optional[Literal["sigmoid", "softmax"]] = "sigmoid",
) -> tuple[Tensor, Tensor]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    preds = normalize_logits_if_needed(preds, normalization)

    probs_zero_class = torch.ones(preds.shape) - preds
    preds = torch.cat([probs_zero_class.unsqueeze(dim=-1), preds.unsqueeze(dim=-1)], dim=-1)

    return preds, target


def _multiclass_brier_format(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies softmax if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.transpose(0, 1).reshape(num_classes, -1).T
    target = target.flatten()

    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    preds = normalize_logits_if_needed(preds, "softmax")

    return preds, target


def _brier_binary_validation():
    return
