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
from typing import Union

import torch
from torch import Tensor, linalg

from torchmetrics.utilities.checks import _check_same_shape


def procrustes_disparity(
    point_cloud1: Tensor, point_cloud2: Tensor, return_all: bool = False
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """Run Procrustes analysis on a batch of data points.

    Works similar ``scipy.spatial.procrustes`` but for batches of data points.

    Args:
        point_cloud1: The first set of data points
        point_cloud2: The second set of data points
        return_all: If True, returns the scale and rotation matrices along with the disparity

    Raises:
        ValueError: If a point cloud has empty spatial dimensions, non-finite values, or no distinct points.

    """
    _check_same_shape(point_cloud1, point_cloud2)
    if point_cloud1.ndim != 3:
        raise ValueError(
            "Expected both datasets to be 3D tensors of shape (N, M, D), where N is the batch size, M is the number of"
            f" data points and D is the dimensionality of the data points, but got {point_cloud1.ndim} dimensions."
        )

    if point_cloud1.shape[1] == 0 or point_cloud1.shape[2] == 0:
        raise ValueError("Expected point clouds with non-empty point and coordinate dimensions.")
    if not torch.isfinite(point_cloud1).all() or not torch.isfinite(point_cloud2).all():
        raise ValueError("Expected point clouds containing only finite values.")

    point_cloud1 = point_cloud1 - point_cloud1.mean(dim=1, keepdim=True)
    point_cloud2 = point_cloud2 - point_cloud2.mean(dim=1, keepdim=True)
    norm1 = linalg.norm(point_cloud1, dim=[1, 2], keepdim=True)
    norm2 = linalg.norm(point_cloud2, dim=[1, 2], keepdim=True)
    if (norm1 == 0).any() or (norm2 == 0).any():
        raise ValueError("Expected each point cloud to contain more than one unique point.")
    if not torch.isfinite(norm1).all() or not torch.isfinite(norm2).all():
        raise ValueError("Expected point clouds with finite centered norms.")
    point_cloud1 = point_cloud1 / norm1
    point_cloud2 = point_cloud2 / norm2

    u, w, v = linalg.svd(torch.matmul(point_cloud2.transpose(1, 2), point_cloud1).transpose(1, 2), full_matrices=False)

    rotation = torch.matmul(u, v)
    scale = w.sum(1, keepdim=True)
    point_cloud2 = scale[:, None] * torch.matmul(point_cloud2, rotation.transpose(1, 2))
    disparity = (point_cloud1 - point_cloud2).square().sum(dim=[1, 2])
    if return_all:
        return disparity, scale, rotation
    return disparity
