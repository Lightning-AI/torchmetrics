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
from typing import Optional

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
from torchmetrics.utilities.compute import _safe_matmul


def _pairwise_linear_similarity_update(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculate the pairwise linear similarity matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero
    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)

    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance


def pairwise_linear_similarity(
    x: Tensor,
    y: Optional[Tensor] = None,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> Tensor:
    r"""Calculate pairwise linear similarity.

    .. math::
        s_{lin}(x,y) = <x,y> = \sum_{d=1}^D x_d \cdot y_d

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between
    the rows of :math:`x` and :math:`y`.
    If only :math:`x` is passed in, the calculation will be performed between the rows of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional.pairwise import pairwise_linear_similarity
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_linear_similarity(x, y)
        tensor([[ 2.,  7.],
                [ 3., 11.],
                [ 5., 18.]])
        >>> pairwise_linear_similarity(x)
        tensor([[ 0., 21., 34.],
                [21.,  0., 55.],
                [34., 55.,  0.]])
    """
    distance = _pairwise_linear_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
