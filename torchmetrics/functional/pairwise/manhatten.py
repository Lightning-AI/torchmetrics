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

from deprecate import deprecated, void
from torch import Tensor

from torchmetrics.functional.pairwise.manhattan import pairwise_manhattan_distance


@deprecated(target=pairwise_manhattan_distance, deprecated_in="0.7", remove_in="0.8")
def pairwise_manhatten_distance(
    x: Tensor, y: Optional[Tensor] = None, reduction: Optional[str] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    r"""
    Calculates pairwise manhattan distance.

    .. deprecated:: v0.7
        Use :func:`torchmetrics.functional.pairwise_manhattan_distance`. Will be removed in v0.8.

    Example:
        >>> import torch
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
    return void(x, y, reduction, zero_diagonal)
