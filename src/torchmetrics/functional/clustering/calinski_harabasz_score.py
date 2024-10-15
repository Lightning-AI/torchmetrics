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
    _validate_intrinsic_cluster_data,
    _validate_intrinsic_labels_to_samples,
)


def calinski_harabasz_score(data: Tensor, labels: Tensor) -> Tensor:
    """Compute the Calinski Harabasz Score (also known as variance ratio criterion) for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Calinski Harabasz Score

    Example:
        >>> from torch import randn, randint
        >>> from torchmetrics.functional.clustering import calinski_harabasz_score
        >>> data = randn(20, 3)
        >>> labels = randint(0, 3, (20,))
        >>> calinski_harabasz_score(data, labels)
        tensor(2.2128)

    """
    _validate_intrinsic_cluster_data(data, labels)

    # convert to zero indexed labels
    unique_labels, labels = torch.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    num_samples = data.shape[0]
    _validate_intrinsic_labels_to_samples(num_labels, num_samples)

    mean = data.mean(dim=0)
    between_cluster_dispersion = torch.tensor(0.0, device=data.device)
    within_cluster_dispersion = torch.tensor(0.0, device=data.device)
    for k in range(num_labels):
        cluster_k = data[labels == k, :]
        mean_k = cluster_k.mean(dim=0)
        between_cluster_dispersion += ((mean_k - mean) ** 2).sum() * cluster_k.shape[0]
        within_cluster_dispersion += ((cluster_k - mean_k) ** 2).sum()

    if within_cluster_dispersion == 0:
        return torch.tensor(1.0, device=data.device, dtype=torch.float32)
    return between_cluster_dispersion * (num_samples - num_labels) / (within_cluster_dispersion * (num_labels - 1.0))
