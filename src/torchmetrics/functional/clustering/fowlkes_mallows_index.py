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


def _fowlkes_mallows_index_update(preds: Tensor, target: Tensor) -> tuple[Tensor, int]:
    """Return contingency matrix required to compute the Fowlkes-Mallows index.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target), preds.size(0)


def _fowlkes_mallows_index_compute(contingency: Tensor, n: int) -> Tensor:
    """Compute the Fowlkes-Mallows index based on the contingency matrix.

    Args:
        contingency: contingency matrix
        n: number of samples

    Returns:
        fowlkes_mallows: Fowlkes-Mallows index

    """
    tk = torch.sum(contingency**2) - n
    if torch.allclose(tk, tensor(0)):
        return torch.tensor(0.0, device=contingency.device)

    pk = torch.sum(contingency.sum(dim=0) ** 2) - n
    qk = torch.sum(contingency.sum(dim=1) ** 2) - n

    return torch.sqrt(tk / pk) * torch.sqrt(tk / qk)


def fowlkes_mallows_index(preds: Tensor, target: Tensor) -> Tensor:
    """Compute Fowlkes-Mallows index between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with Fowlkes-Mallows index

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering import fowlkes_mallows_index
        >>> preds = torch.tensor([2, 2, 0, 1, 0])
        >>> target = torch.tensor([2, 2, 1, 1, 0])
        >>> fowlkes_mallows_index(preds, target)
        tensor(0.5000)

    """
    contingency, n = _fowlkes_mallows_index_update(preds, target)
    return _fowlkes_mallows_index_compute(contingency, n)
