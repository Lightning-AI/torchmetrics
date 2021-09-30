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

from torchmetrics.functional.pairwise.euclidean import _check_input, _pairwise_euclidean_distance_compute


def _pairwise_manhatten_distance_update(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = "mean", zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculates the pairwise manhatten similarity matrix.

    Args:
        X: tensor of shape ``[N,d]``
        Y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal should be set to zero
    """
    X, Y, zero_diagonal = _check_input(X, Y, zero_diagonal)

    distance = (X.unsqueeze(1) - Y.unsqueeze(0).repeat(X.shape[0], 1, 1)).abs().sum(dim=-1)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance


def pairwise_manhatten_distance(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    r"""
    Calculates pairwise manhatten distance:

    .. math::
        d_{man}(x,y) = ||x-y||_1 = \sum_{d=1}^D |x_d - y_d|

    If two tensors are passed in, the calculation will be performed
    pairwise between the rows of the tensors. If a single tensor is passed in, the calculation will
    be performed between the rows of that tensor.

    Args:
        X: Tensor with shape ``[N, d]``
        Y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `X` is given
            this defaults to `True` else if `Y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``X`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional import pairwise_manhatten_distance
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_manhatten_distance(x, y)
        tensor([[ 4.,  2.],
                [ 7.,  5.],
                [12., 10.]])
        >>> pairwise_manhatten_distance(x)
        tensor([[0., 3., 8.],
                [3., 0., 5.],
                [8., 5., 0.]])

    """
    distance = _pairwise_manhatten_distance_update(X, Y, zero_diagonal)
    return _pairwise_euclidean_distance_compute(distance, reduction)
