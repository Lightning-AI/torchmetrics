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
from typing import Tuple, Union

import torch
from torch import Tensor, linalg

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.prints import rank_zero_warn


def procrustes_disparity(
    dataset1: Tensor, dataset2: Tensor, return_all: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Runs procrustrus analysis on a batch of data points.

    Works similar ``scipy.spatial.procrustes`` but for batches of data points.

    Args:
        dataset1: The first set of data points
        dataset2: The second set of data points
        return_all: If True, returns the scale and rotation matrices along with the disparity

    """
    _check_same_shape(dataset1, dataset2)
    if dataset1.ndim != 3:
        raise ValueError(
            "Expected both datasets to be 3D tensors of shape (N, M, D), where N is the batch size, M is the number of"
            f" data points and D is the dimensionality of the data points, but got {dataset1.ndim} dimensions."
        )

    dataset1 = dataset1 - dataset1.mean(dim=1, keepdim=True)
    dataset2 = dataset2 - dataset2.mean(dim=1, keepdim=True)
    dataset1 /= linalg.norm(dataset1, dim=[1, 2], keepdim=True)
    dataset2 /= linalg.norm(dataset2, dim=[1, 2], keepdim=True)

    try:
        u, w, v = linalg.svd(torch.matmul(dataset2.transpose(1, 2), dataset1).transpose(1, 2), full_matrices=False)
    except Exception as ex:
        rank_zero_warn(
            f"SVD calculation in procrustes_disparity failed with exception {ex}. Returning 0 disparity and identity"
            " scale/rotation.",
            UserWarning,
        )
        return torch.tensor(0.0), torch.ones(dataset1.shape[0]), torch.eye(dataset1.shape[2])

    rotation = torch.matmul(u, v)
    scale = w.sum(1, keepdim=True)
    dataset2 = scale[:, None] * torch.matmul(dataset2, rotation.transpose(1, 2))
    disparity = (dataset1 - dataset2).square().sum(dim=[1, 2])
    if return_all:
        return disparity, scale, rotation
    return disparity
