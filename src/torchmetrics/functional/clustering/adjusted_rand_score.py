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
    calculate_pair_cluster_confusion_matrix,
    check_cluster_labels,
)


def _adjusted_rand_score_update(preds: Tensor, target: Tensor) -> Tensor:
    """Update and return variables required to compute the rand score.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _adjusted_rand_score_compute(contingency: Tensor) -> Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    (tn, fp), (fn, tp) = calculate_pair_cluster_confusion_matrix(contingency=contingency)
    if fn == 0 and fp == 0:
        return torch.ones_like(tn, dtype=torch.float32)
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def adjusted_rand_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the Adjusted Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with adjusted rand score

    Example:
        >>> from torchmetrics.functional.clustering import adjusted_rand_score
        >>> import torch
        >>> adjusted_rand_score(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
        tensor(1.)
        >>> adjusted_rand_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.5714)

    """
    contingency = _adjusted_rand_score_update(preds, target)
    return _adjusted_rand_score_compute(contingency)
