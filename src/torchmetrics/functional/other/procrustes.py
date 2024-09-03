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
from torch import Tensor, linalg

def procrustes_disparity(data1: Tensor, data2: Tensor, return_all: bool = False) -> Tensor:
    """Runs procrustrus analysis on a batch of data points.

    Args:
        data1: The first set of data points
        data2: The second set of data points
    
    """
    if data1.shape != data2.shape:
        raise ValueError("data1 and data2 must have the same shape")
    if data1.ndim == 2:
        data1 = data1[None, :, :]
        data2 = data2[None, :, :]

    data1 -= data1.mean(dim=1, keepdim=True)
    data2 -= data2.mean(dim=1, keepdim=True)
    data1 /= linalg.norm(data1, dim=[1,2], keepdim=True)
    data2 /= linalg.norm(data2, dim=[1,2], keepdim=True)

    try:
        u, w, v = linalg.svd(torch.matmul(data2.transpose(1, 2), data1).transpose(1,2), full_matrices=False)
    except:
        raise ValueError("SVD did not converge")
    rotation = torch.matmul(u, v)
    scale = w.sum(1, keepdim=True)
    data2 = scale[:,None] * torch.matmul(data2, rotation.transpose(1,2))
    disparity = (data1 - data2).square().sum(dim=[1,2])
    if return_all:
        return disparity, scale, rotation
    return disparity

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html