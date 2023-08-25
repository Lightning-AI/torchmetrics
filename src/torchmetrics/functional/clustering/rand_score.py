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

from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels


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
    n_samples = contingency.sum()
    n_c = contingency.sum(dim=1)
    n_k = contingency.sum(dim=0)
    sum_squared = (contingency**2).sum()

    pair_matrix = torch.zeros(2, 2, dtype=contingency.dtype, device=contingency.device)
    pair_matrix[1, 1] = sum_squared - n_samples
    pair_matrix[0, 1] = (contingency * n_k).sum() - sum_squared
    pair_matrix[1, 0] = (contingency.T * n_c).sum() - sum_squared
    pair_matrix[0, 0] = n_samples**2 - pair_matrix[0, 1] - pair_matrix[1, 0] - sum_squared

    numerator = pair_matrix.diagonal().sum()
    denominator = pair_matrix.sum()
    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0

    return numerator / denominator


def rand_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        rand_score: rand score

    """
    contingency = _rand_score_update(preds, target)
    return _rand_score_compute(contingency)
