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
from typing import Tuple

import torch
from torch import Tensor, tensor

from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels


def _mutual_info_score_update(
    preds: Tensor,
    target: Tensor,
    # num_classes: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update and return variables required to compute the mutual information score.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _mutual_info_score_compute(contingency: Tensor) -> Tensor:
    """Compute the mutual information score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        mutual_info: mutual information score

    """
    N = contingency.sum()
    U = contingency.sum(dim=1)
    V = contingency.sum(dim=0)

    # Check if preds or target labels only have one cluster
    if U.size() == 1 or V.size() == 1:
        return tensor(0.0)

    log_outer = torch.log(U).reshape(-1, 1) + torch.log(V)
    mutual_info = contingency / N * (torch.log(N) + torch.log(contingency) - log_outer)
    return mutual_info.sum()


def mutual_info_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute mutual information between two clusterings.

    Args:
        preds: predicted classes
        target: ground truth classes

    Example:
        >>> from torchmetrics.functional.clustering import mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> mutual_info_score(preds, target)
        tensor([1.05492])

    """
    contingency = _mutual_info_score_update(preds, target)
    return _mutual_info_score_compute(contingency)
