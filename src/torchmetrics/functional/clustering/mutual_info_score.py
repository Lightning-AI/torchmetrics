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

from typing import Optional, Tuple
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_same_shape


def check_cluster_labels(preds: Tensor, target: Tensor) -> None:
    """Check shape of input tensors."""
    _check_same_shape(preds, target)
    if torch.is_floating_point(preds) or torch.is_floating_point(target):
        raise ValueError(
            f"Expected discrete values but received {preds.dtype} for"
            f"predictions and {target.dtype} for target labels instead."
        )


def _calculate_contingency_matrix(
    preds: Tensor,
    target: Tensor,
    eps: Optional[float] = 1e-16,
    sparse: bool = False
) -> Tensor:
    """Calculate contingency matrix.

    Args:
        preds: predicted labels
        target: ground truth labels
        sparse: If True, returns contingency matrix as a sparse matrix.

    Returns:
        contingency: contingency matrix of shape (n_classes_target, n_classes_preds)

    """
    if eps is not None and sparse is True:
        raise ValueError('Cannot specify `eps` and return sparse tensor.')

    preds_classes, preds_idx = torch.unique(preds, return_inverse=True)
    target_classes, target_idx = torch.unique(target, return_inverse=True)

    n_classes_preds = preds_classes.size(0)
    n_classes_target = target_classes.size(0)

    contingency = torch.sparse_coo_tensor(
        torch.stack((target_idx, preds_idx)),
        torch.ones(target_idx.size(0)),
        (n_classes_target, n_classes_preds)
    )

    if not sparse:
        contingency = contingency.to_dense()
    if eps:
        contingency = contingency + eps

    return contingency


def _mutual_info_score_update(
    preds: Tensor,
    target: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update and return variables required to compute the mutual information score.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return _calculate_contingency_matrix(preds, target)


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
