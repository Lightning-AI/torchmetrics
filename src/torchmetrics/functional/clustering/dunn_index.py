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
from itertools import combinations

import torch
from torch import Tensor


def _dunn_index_update(data: Tensor, labels: Tensor, p: float) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute the Dunn index.

    Args:
        data: feature vectors of shape (n_samples, n_features)
        labels: cluster labels
        p: p-norm (distance metric)

    Returns:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    """
    unique_labels, inverse_indices = labels.unique(return_inverse=True)
    clusters = [data[inverse_indices == label_idx] for label_idx in range(len(unique_labels))]
    centroids = [c.mean(dim=0) for c in clusters]

    intercluster_distance = torch.linalg.norm(
        torch.stack([a - b for a, b in combinations(centroids, 2)], dim=0), ord=p, dim=1
    )

    max_intracluster_distance = torch.stack([
        torch.linalg.norm(ci - mu, ord=p, dim=1).max() for ci, mu in zip(clusters, centroids)
    ])

    return intercluster_distance, max_intracluster_distance


def _dunn_index_compute(intercluster_distance: Tensor, max_intracluster_distance: Tensor) -> Tensor:
    """Compute the Dunn index based on updated state.

    Args:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    Returns:
        scalar tensor with the dunn index

    """
    return intercluster_distance.min() / max_intracluster_distance.max()


def dunn_index(data: Tensor, labels: Tensor, p: float = 2) -> Tensor:
    """Compute the Dunn index.

    Args:
        data: feature vectors
        labels: cluster labels
        p: p-norm used for distance metric

    Returns:
        scalar tensor with the dunn index

    Example:
        >>> from torchmetrics.functional.clustering import dunn_index
        >>> data = torch.tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        >>> labels = torch.tensor([0, 0, 0, 1])
        >>> dunn_index(data, labels)
        tensor(2.)

    """
    pairwise_distance, max_distance = _dunn_index_update(data, labels, p)
    return _dunn_index_compute(pairwise_distance, max_distance)
