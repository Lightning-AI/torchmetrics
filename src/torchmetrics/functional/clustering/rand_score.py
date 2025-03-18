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


def _rand_score_update(preds: Tensor, target: Tensor) -> Tensor:
    """Update and return variables required to compute the rand score.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _rand_score_compute(contingency: Tensor) -> Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    pair_matrix = calculate_pair_cluster_confusion_matrix(contingency=contingency)

    numerator = pair_matrix.diagonal().sum()
    denominator = pair_matrix.sum()
    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return torch.ones_like(numerator, dtype=torch.float32)

    return numerator / denominator


def rand_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from torchmetrics.functional.clustering import rand_score
        >>> import torch
        >>> rand_score(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> rand_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.8333)

    """
    contingency = _rand_score_update(preds, target)
    return _rand_score_compute(contingency)
