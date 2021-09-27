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

import torch
from torch import Tensor

from torchmetrics.functional.pairwise.euclidean import (
    _pairwise_euclidean_distance_compute, 
    _check_input
)


def _pairwise_manhatten_distance_update(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = 'mean', zero_diagonal: Optional[bool] = None
) -> Tensor:
    X, Y, zero_diagonal = _check_input(X, Y, zero_diagonal)

    distance = (X.unsqueeze(1) - Y.unsqueeze(0).repeat(X.shape[0], 1, 1)).abs().sum(dim=-1)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance


def pairwise_manhatten_distance(
    X: Tensor, Y: Optional[Tensor] = None, reduction: Optional[str] = 'mean', zero_diagonal: Optional[bool] = None
) -> Tensor:
    """

    """
    distance = _pairwise_manhatten_distance_update(X, Y, zero_diagonal)
    return _pairwise_euclidean_distance_compute(distance, reduction)
