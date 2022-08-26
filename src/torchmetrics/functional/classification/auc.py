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
import torch
from torch import Tensor

from torchmetrics.utilities.compute import auc as _auc
from torchmetrics.utilities.prints import rank_zero_warn


def auc(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Computes Area Under the Curve (AUC) using the trapezoidal rule.

    .. note::
        This metric have been moved to `torchmetrics.utilities.compute` in v0.10 this version will be removed in v0.11.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Return:
        Tensor containing AUC score

    Raises:
        ValueError:
            If both ``x`` and ``y`` tensors are not ``1d``.
        ValueError:
            If both ``x`` and ``y`` don't have the same numnber of elements.
        ValueError:
            If ``x`` tesnsor is neither increasing nor decreasing.

    Example:
        >>> from torchmetrics.functional import auc
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auc(x, y)
        tensor(4.)
        >>> auc(x, y, reorder=True)
        tensor(4.)
    """
    rank_zero_warn(
        "`torchmetrics.functional.auc` has been move to `torchmetrics.utilities.compute` in v0.10"
        " and will be removed in v0.11.",
        DeprecationWarning,
    )
    return _auc(x, y, reorder=reorder)
