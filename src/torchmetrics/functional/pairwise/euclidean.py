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

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix


def _pairwise_euclidean_distance_update(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculate the pairwise euclidean distance matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero
    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    # upcast to float64 to prevent precision issues
    _orig_dtype = x.dtype
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1)
    distance = (x_norm + y_norm - 2 * x.mm(y.T)).to(_orig_dtype)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.sqrt()


def pairwise_euclidean_distance(
    x: Tensor,
    y: Optional[Tensor] = None,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> Tensor:
    r"""Calculate pairwise euclidean distances.

    .. math::
        d_{euc}(x,y) = ||x - y||_2 = \sqrt{\sum_{d=1}^D (x_d - y_d)^2}

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
        >>> from torchmetrics.functional.pairwise import pairwise_euclidean_distance
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
    distance = _pairwise_euclidean_distance_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
