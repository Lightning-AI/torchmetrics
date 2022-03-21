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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.metric import Metric


class RootMeanSquaredErrorUsingSlidingWindow(Metric):
    """Computes Root Mean Squared Error (RMSE) using sliding window.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        return_rmse_map: An indication whether

    Return:
        RMSE using sliding window
        (Optionally) RMSE map

    Example:
        >>> from torchmetrics import RootMeanSquaredErrorUsingSlidingWindow
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> rmse_sw = RootMeanSquaredErrorUsingSlidingWindow()
        >>> rmse_sw(preds, target)
        tensor(0.4008)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.
    """

    rmse_val_sum: Tensor
    rmse_map: Tensor = None
    total_images: Tensor
    higher_is_better = False

    def __init__(
        self,
        window_size: int = 8,
        return_rmse_map: bool = False,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.add_state("rmse_val_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("add_total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if not isinstance(window_size, int) or isinstance(window_size, int) and window_size < 1:
            raise ValueError("Argument `window_size` is expected to be a positive integer.")

        self.window_size = window_size
        self.return_rmse_map = return_rmse_map

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Updates intermediate rmse values and map.

        Args:
            preds: Deformed image
            target: Ground truth image
        """
        if self.return_rmse_map and self.rmse_map is None:
            _img_shape = target.shape[1:]  # channels, width, height
            self.rmse_map = torch.zeros(_img_shape, dtype=target.dtype, device=target.device)

        self.rmse_val_sum, self.rmse_map, self.total_images = _rmse_sw_update(
            preds, target, self.window_size, self.rmse_val_sum, self.rmse_map, self.add_total_images
        )

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Computes Root Mean Squared Error (using sliding window) and potentially return RMSE map."""
        rmse, rmse_map = _rmse_sw_compute(self.rmse_val_sum, self.rmse_map, self.total_images)
        if self.return_rmse_map:
            return rmse, rmse_map
        return rmse
