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
import torch
from torch import Tensor

from torchmetrics.functional.clustering.utils import (
    calculate_contingency_matrix,
    check_cluster_labels,
    calculate_entropy
)
from torchmetrics.functional.clustering.mutual_info_score import _mutual_info_score_compute


def _homogeneity_score_compute(preds: Tensor, target: Tensor) -> Tensor:
    """ Computes the homogeneity score of a clustering given the predicted and target cluster labels."""
    check_cluster_labels(preds, target)

    if len(target) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=preds.device)

    entropy_target = calculate_entropy(target)
    entropy_preds = calculate_entropy(preds)

    contingency = calculate_contingency_matrix(preds, target)
    mutual_info = _mutual_info_score_compute(contingency)

    homogeneity = mutual_info / entropy_target if entropy_target else torch.ones_like(entropy_target)
    return homogeneity, mutual_info, entropy_preds, entropy_target


def _completeness_score_compute(preds: Tensor, target: Tensor) -> Tensor:
    """ Computes the completeness score of a clustering given the predicted and target cluster labels."""
    homogeneity, mutual_info, entropy_preds, _ = _homogeneity_score_compute(preds, target)
    completeness = mutual_info / entropy_preds if entropy_preds else torch.ones_like(entropy_preds)
    return completeness, homogeneity


def completeness_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from torchmetrics.functional.clustering import completeness_score
        >>> import torch
        >>> completeness_score(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> completeness_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.8333)

    """
    completeness, _ = _completeness_score_compute(preds, target)
    return completeness


def homogeneity_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from torchmetrics.functional.clustering import homogeneity_score
        >>> import torch
        >>> homogeneity_score(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> homogeneity_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.8333)

    """
    homogeneity, _, _, _ = _homogeneity_score_compute(preds, target)
    return homogeneity


def v_measure_score(preds: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from torchmetrics.functional.clustering import v_score
        >>> import torch
        >>> v_score(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> v_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.8333)

    """
    homogeneity, completeness = _completeness_score_compute(preds, target)
    if homogeneity + completeness == 0.0:
        v_measure = torch.ones_like(homogeneity)
    else:
        v_measure = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    return v_measure


