# Copyright The PyTorch Lightning team.
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
from typing import Optional

from torch import Tensor


def _pairwise_euclidean_distance_update(
    X: Tensor, Y: Optional[Tensor] = None, zero_diagonal: bool = True
) -> Tensor:
    if X.ndim != 2:
        raise ValueError('Expected argument `X` to be a 2D tensor of shape `[N, d]`')

    if Y is not None:
        if Y.ndim != 2 or Y.shape[1] != X.shape[1]:
            raise ValueError('Expected argument `Y` to be a 2D tensor of shape `[M, d]` where'
                             ' `d` should be same as the last dimension of `X`')

        distance = X.norm(dim=1, keepdim=True)**2 + Y.norm(dim=1).T**2 - 2*X.mm(Y.T)
    else:
        distance = X.mm(X)
        distance = distance.fill_diagonal_(0) if zero_diagonal else distance
    
    return distance


def _pairwise_euclidean_distance_compute(distance: Tensor, reduction: Tensor) -> Tensor:
    if reduction == 'mean':
        return distance.mean(dim=-1)
    elif reduction == 'sum':
        return distance.sum(dim=-1)
    elif reduction is None:
        return distance
    else:
        raise ValueError(f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}")


def pairwise_euclidean_distance(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = 'mean', zero_diagonal: bool = True
) -> Tensor:
    distance = _pairwise_euclidean_distance_update(X, Y, zero_diagonal)
    return _pairwise_euclidean_distance_compute(distance, reduction)

