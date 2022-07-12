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

from torchmetrics.functional.image.tv import _total_variation_compute, _total_variation_update
from torchmetrics.metric import Metric


class TotalVariation(Metric):
    """Computes Total Variation loss (`TV`_).

    Adapted from: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html

    Args:
        reduction: a method to reduce metric score over samples.
            - ``'mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'`` or ``'mean'``

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
    current: torch.Tensor
    total: torch.Tensor

    def __init__(self, reduction: str = "sum", **kwargs):
        super().__init__(**kwargs)
        if reduction not in ("sum", "mean"):
            raise ValueError("Expected argument `reduction` to either be 'sum' or 'mean'")
        self.reduction = reduction

        self.add_state("score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, img: torch.Tensor) -> None:
        """Update current score with batch of input images.

        Args:
            img: A `torch.Tensor` of shape `(N, C, H, W)` consisting of images
        """
        score, num_elements = _total_variation_update(img)
        self.score += score
        self.num_elements += num_elements

    def compute(self):
        """Compute final total variation."""
        return _total_variation_compute(self.score, self.num_elements, self.reduction)
