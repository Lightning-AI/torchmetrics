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


def _check_input(
    X: Tensor, Y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Union[Tensor, Tensor, bool]:
    """Check that input has the right dimensionality and sets the zero_diagonal argument if user has not provided
    import module.

    Args:
        X: tensor of shape ``[N,d]``
        Y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal should be set to zero
    """
    if X.ndim != 2:
        raise ValueError(f"Expected argument `X` to be a 2D tensor of shape `[N, d]` but got {X.shape}")

    if Y is not None:
        if Y.ndim != 2 or Y.shape[1] != X.shape[1]:
            raise ValueError(
                "Expected argument `Y` to be a 2D tensor of shape `[M, d]` where"
                " `d` should be same as the last dimension of `X`"
            )
        zero_diagonal = False if zero_diagonal is None else zero_diagonal
    else:
        Y = X.clone()
        zero_diagonal = True if zero_diagonal is None else zero_diagonal
    return X, Y, zero_diagonal


def _pairwise_euclidean_distance_update(
    X: Tensor, Y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculates the pairwise euclidean distance matrix.

    Args:
        X: tensor of shape ``[N,d]``
        Y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal should be set to zero
    """
    X, Y, zero_diagonal = _check_input(X, Y, zero_diagonal)

    distance = X.norm(dim=1, keepdim=True) ** 2 + Y.norm(dim=1).T ** 2 - 2 * X.mm(Y.T)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.sqrt()


def _pairwise_euclidean_distance_compute(distance: Tensor, reduction: Tensor) -> Tensor:
    """Final reduction of distance matrix.

    Args:
        distance: a ``[N,M]`` matrix
        reduction: string determining how to reduce along last dimension
    """
    if reduction == "mean":
        return distance.mean(dim=-1)
    if reduction == "sum":
        return distance.sum(dim=-1)
    if reduction is None or reduction == "none":
        return distance
    raise ValueError(f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}")


def pairwise_euclidean_distance(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    r"""
    Calculates pairwise euclidean distances:

    .. math::
        d_{euc}(x,y) = ||x - y||_2 = \sqrt{\sum_{d=1}^D (x_d - y_d)^2}

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
        >>> from torchmetrics.functional import pairwise_euclidean_distance
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_euclidean_distance(x, y)
        tensor([[3.1623, 2.0000],
                [5.3852, 4.1231],
                [8.9443, 7.6158]])
        >>> pairwise_euclidean_distance(x)
        tensor([[0.0000, 2.2361, 5.8310],
                [2.2361, 0.0000, 3.6056],
                [5.8310, 3.6056, 0.0000]])

    """
    distance = _pairwise_euclidean_distance_update(X, Y, zero_diagonal)
    return _pairwise_euclidean_distance_compute(distance, reduction)
