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


def davies_bouldin_score(data: Tensor, labels: Tensor) -> Tensor:
    """Compute the Davies bouldin score for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Davies bouldin score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering import davies_bouldin_score
        >>> _ = torch.manual_seed(42)
        >>> data = torch.randn(10, 3)
        >>> labels = torch.randint(0, 2, (10,))
        >>> davies_bouldin_score(data, labels)
        tensor(1.3249)

    """
    _validate_intrinsic_cluster_data(data, labels)

    # convert to zero indexed labels
    unique_labels, labels = torch.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    num_samples, dim = data.shape
    _validate_intrinsic_labels_to_samples(num_labels, num_samples)

    intra_dists = torch.zeros(num_labels, device=data.device)
    centroids = torch.zeros((num_labels, dim), device=data.device)
    for k in range(num_labels):
        cluster_k = data[labels == k, :]
        centroids[k] = cluster_k.mean(dim=0)
        intra_dists[k] = (cluster_k - centroids[k]).pow(2.0).sum(dim=1).sqrt().mean()
    centroid_distances = torch.cdist(centroids, centroids)

    cond1 = torch.allclose(intra_dists, torch.zeros_like(intra_dists))
    cond2 = torch.allclose(centroid_distances, torch.zeros_like(centroid_distances))
    if cond1 or cond2:
        return torch.tensor(0.0, device=data.device, dtype=torch.float32)

    centroid_distances[centroid_distances == 0] = float("inf")
    combined_intra_dists = intra_dists.unsqueeze(0) + intra_dists.unsqueeze(1)
    scores = (combined_intra_dists / centroid_distances).max(dim=1).values
    return scores.mean()
