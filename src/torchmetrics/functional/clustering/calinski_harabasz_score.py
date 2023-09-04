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


def _calinski_harabasz_score_validate_input(data: Tensor, labels: Tensor) -> None:
    """Validate that the input data and labels have correct shape and type."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got {data.ndim}D data instead")
    if not data.is_floating_point():
        raise ValueError(f"Expected floating point data, got {data.dtype} data instead")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got {labels.ndim}D labels instead")


def calinski_harabasz_score(data: Tensor, labels: Tensor) -> Tensor:
    """Compute the Calinski Harabasz Score (also known as variance ratio criterion) for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Calinski Harabasz Score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering import calinski_harabasz_score
        >>> _ = torch.manual_seed(42)
        >>> data = torch.randn(10, 3)
        >>> labels = torch.randint(0, 2, (10,))
        >>> calinski_harabasz_score(data, labels)
        tensor(3.4998)

    """
    _calinski_harabasz_score_validate_input(data, labels)

    # convert to zero indexed labels
    unique_labels, labels = torch.unique(labels, return_inverse=True)
    n_labels = len(unique_labels)

    n_samples = data.shape[0]

    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of detected clusters must be greater than one and less than the number of samples."
            f"Got {n_labels} clusters and {n_samples} samples."
        )

    mean = data.mean(dim=0)
    between_cluster_dispersion = torch.tensor(0.0, device=data.device)
    within_cluster_dispersion = torch.tensor(0.0, device=data.device)
    for k in range(n_labels):
        cluster_k = data[labels == k, :]
        mean_k = cluster_k.mean(dim=0)
        between_cluster_dispersion += ((mean_k - mean) ** 2).sum() * cluster_k.shape[0]
        within_cluster_dispersion += ((cluster_k - mean_k) ** 2).sum()

    if within_cluster_dispersion == 0:
        return torch.tensor(1.0, device=data.device, dtype=torch.float32)
    return between_cluster_dispersion * (n_samples - n_labels) / (within_cluster_dispersion * (n_labels - 1.0))
