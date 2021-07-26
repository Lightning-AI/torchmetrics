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
from warnings import warn

import torch
from torch import Tensor

from torchmetrics.functional.regression.r2 import r2_score


def r2score(
    preds: Tensor,
    target: Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> Tensor:
    r"""
    Computes r2 score also known as `coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

    .. deprecated:: v0.5
        `r2score` was renamed as `r2_score` in v0.5 and it will be removed in v0.6

    Example:
        >>> from torchmetrics.functional import r2score
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> r2score(preds, target)
        tensor(0.9486)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2score(preds, target, multioutput='raw_values')
        tensor([0.9654, 0.9082])
    """
    warn("`r2score` was renamed as `r2_score` in v0.5 and it will be removed in v0.6", DeprecationWarning)
    return r2_score(preds, target, adjusted, multioutput)
