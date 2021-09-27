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
from typing import Optional, Union

from torch import Tensor

def _check_input(X: Tensor, Y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None) -> Union[Tensor, Tensor, bool]:
    if X.ndim != 2:
        raise ValueError(f'Expected argument `X` to be a 2D tensor of shape `[N, d]` but got {X.shape}')

    if Y is not None:
        if Y.ndim != 2 or Y.shape[1] != X.shape[1]:
            raise ValueError('Expected argument `Y` to be a 2D tensor of shape `[M, d]` where'
                             ' `d` should be same as the last dimension of `X`')
        if zero_diagonal is None:
            zero_diagonal = False
    else:
        Y = X.clone()
        if zero_diagonal is None:
            zero_diagonal = True
    return X, Y, zero_diagonal


def _pairwise_euclidean_distance_update(
    X: Tensor, Y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    X, Y, zero_diagonal = _check_input(X, Y, zero_diagonal)

    distance = X.norm(dim=1, keepdim=True)**2 + Y.norm(dim=1).T**2 - 2 * X.mm(Y.T)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.sqrt()


def _pairwise_euclidean_distance_compute(distance: Tensor, reduction: Tensor) -> Tensor:
    if reduction == 'mean':
        return distance.mean(dim=-1)
    elif reduction == 'sum':
        return distance.sum(dim=-1)
    elif reduction is None or reduction == 'none':
        return distance
    else:
        raise ValueError(f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}")


def pairwise_euclidean_distance(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = 'mean', zero_diagonal: Optional[bool] = None
) -> Tensor:
    """ Calculates pairwise distances. If two tensors are passed in, the calculation will be performed
        pairwise between the rows of the tensors. If a single tensor is passed in, the calculation will
        be performed between the rows of that tensor.
    
    Args:
        X: Tensor with shape ``[N, d]``
        Y: Tensor with shape ``[M, d]``
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`, `'none'`,
            or `None`
        zero_diagonal: if the diagonal of the distance matrix should be set to 0
    """

    distance = _pairwise_euclidean_distance_update(X, Y, zero_diagonal)
    return _pairwise_euclidean_distance_compute(distance, reduction)

