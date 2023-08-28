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
from torch import Tensor, tensor

from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels


def _mutual_info_score_update(preds: Tensor, target: Tensor) -> Tensor:
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
    n = contingency.sum()
    u = contingency.sum(dim=1)
    v = contingency.sum(dim=0)

    # Check if preds or target labels only have one cluster
    if u.size() == 1 or v.size() == 1:
        return tensor(0.0)

    # Find indices of nonzero values in U and V
    nzu, nzv = torch.nonzero(contingency, as_tuple=True)
    contingency = contingency[nzu, nzv]

    # Calculate MI using entries corresponding to nonzero contingency matrix entries
    log_outer = torch.log(u[nzu]) + torch.log(v[nzv])
    mutual_info = contingency / n * (torch.log(n) + torch.log(contingency) - log_outer)
    return mutual_info.sum()


def mutual_info_score(preds: Tensor, target: Tensor) -> Tensor:
    """Compute mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Example:
        >>> from torchmetrics.functional.clustering import mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> mutual_info_score(preds, target)
        tensor(1.0549)

    """
    contingency = _mutual_info_score_update(preds, target)
    return _mutual_info_score_compute(contingency)
