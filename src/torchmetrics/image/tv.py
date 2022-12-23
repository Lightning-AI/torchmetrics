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
from typing import Any

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.image.tv import _total_variation_compute, _total_variation_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class TotalVariation(Metric):
    """Computes Total Variation loss (`TV`_).

    As input to 'update' the metric accepts the following input:

    - ``img``: A `Tensor` of shape ``(N, C, H, W)`` consisting of images

    Args:
        reduction: a method to reduce metric score over samples

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``

    Example:
        >>> import torch
        >>> from torchmetrics import TotalVariation
        >>> _ = torch.manual_seed(42)
        >>> tv = TotalVariation()
        >>> img = torch.rand(5, 3, 28, 28)
        >>> tv(img)
        tensor(7546.8018)
    """

    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(self, reduction: Literal["mean", "sum", "none", None] = "sum", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction is not None and reduction not in ("sum", "mean", "none"):
            raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")
        self.reduction = reduction

        if self.reduction is None or self.reduction == "none":
            self.add_state("score", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("score", default=tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_elements", default=tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, img: Tensor) -> None:  # type: ignore
        """Update current score with batch of input images."""
        score, num_elements = _total_variation_update(img)
        if self.reduction is None or self.reduction == "none":
            self.score.append(score)
        else:
            self.score += score.sum()
        self.num_elements += num_elements

    def compute(self) -> Tensor:
        """Compute final total variation."""
        if self.reduction is None or self.reduction == "none":
            score = dim_zero_cat(self.score)
        else:
            score = self.score
        return _total_variation_compute(score, self.num_elements, self.reduction)
