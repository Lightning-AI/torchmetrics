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

from typing import Any, Dict, List

import torch
from torch import Tensor

from torchmetrics.functional.image.rase import relative_average_spectral_error
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class RelativeAverageSpectralError(Metric):
    """Computes Relative Average Spectral Error (RASE) (RelativeAverageSpectralError_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``rase`` (:class:`~torch.Tensor`): returns float scalar tensor with average RASE value over sample

    Args:
        window_size: Sliding window used for rmse calculation
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Relative Average Spectral Error (RASE)

    Example:
        >>> from torchmetrics import RelativeAverageSpectralError
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> rase = RelativeAverageSpectralError()
        >>> rase(preds, target)
        tensor(5114.6641)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.
    """

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        window_size: int = 8,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(window_size, int) or isinstance(window_size, int) and window_size < 1:
            raise ValueError(f"Argument `window_size` is expected to be a positive integer, but got {window_size}")
        self.window_size = window_size

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes Relative Average Spectral Error (RASE)."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return relative_average_spectral_error(preds, target, self.window_size)
