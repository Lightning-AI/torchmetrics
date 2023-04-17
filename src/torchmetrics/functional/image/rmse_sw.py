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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.image.helper import _uniform_filter
from torchmetrics.utilities.checks import _check_same_shape


def _rmse_sw_update(
    preds: Tensor,
    target: Tensor,
    window_size: int,
    rmse_val_sum: Optional[Tensor],
    rmse_map: Optional[Tensor],
    total_images: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate the sum of RMSE values and RMSE map for the batch of examples and update intermediate states.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        rmse_val_sum: Sum of RMSE over all examples per individual channels
        rmse_map: Sum of RMSE map values over all examples
        total_images: Total number of images

    Return:
        (Optionally) Intermediate state of RMSE (using sliding window) over the accumulated examples.
        (Optionally) Intermediate state of RMSE map
        Updated total number of already processed images

    Raises:
        ValueError: If ``preds`` and ``target`` do not have the same data type.
        ValueError: If ``preds`` and ``target`` do not have ``BxCxWxH`` shape.
        ValueError: If ``round(window_size / 2)`` is greater or equal to width or height of the image.
    """
    if preds.dtype != target.dtype:
        raise TypeError(
            f"Expected `preds` and `target` to have the same data type. But got {preds.dtype} and {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(f"Expected `preds` and `target` to have BxCxHxW shape. But got {preds.shape}.")

    if round(window_size / 2) >= target.shape[2] or round(window_size / 2) >= target.shape[3]:
        raise ValueError(
            f"Parameter `round(window_size / 2)` is expected to be smaller than {min(target.shape[2], target.shape[3])}"
            f" but got {round(window_size / 2)}."
        )

    if total_images is not None:
        total_images += target.shape[0]
    else:
        total_images = torch.tensor(target.shape[0], device=target.device)
    error = (target - preds) ** 2
    error = _uniform_filter(error, window_size)
    _rmse_map = torch.sqrt(error)
    crop_slide = round(window_size / 2)

    if rmse_val_sum is not None:
        rmse_val = _rmse_map[:, :, crop_slide:-crop_slide, crop_slide:-crop_slide]
        rmse_val_sum += rmse_val.sum(0).mean()
    else:
        rmse_val_sum = _rmse_map[:, :, crop_slide:-crop_slide, crop_slide:-crop_slide].sum(0).mean()

    if rmse_map is not None:
        rmse_map += _rmse_map.sum(0)
    else:
        rmse_map = _rmse_map.sum(0)

    return rmse_val_sum, rmse_map, total_images


def _rmse_sw_compute(
    rmse_val_sum: Optional[Tensor], rmse_map: Tensor, total_images: Tensor
) -> Tuple[Optional[Tensor], Tensor]:
    """Compute RMSE from the aggregated RMSE value. Optionally also computes the mean value for RMSE map.

    Args:
        rmse_val_sum: Sum of RMSE over all examples
        rmse_map: Sum of RMSE map values over all examples
        total_images: Total number of images

    Return:
        RMSE using sliding window
        (Optionally) RMSE map
    """
    rmse = rmse_val_sum / total_images if rmse_val_sum is not None else None
    if rmse_map is not None:
        rmse_map /= total_images
    return rmse, rmse_map


def root_mean_squared_error_using_sliding_window(
    preds: Tensor, target: Tensor, window_size: int = 8, return_rmse_map: bool = False
) -> Union[Optional[Tensor], Tuple[Optional[Tensor], Tensor]]:
    """Compute Root Mean Squared Error (RMSE) using sliding window.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        return_rmse_map: An indication whether the full rmse reduced image should be returned.

    Return:
        RMSE using sliding window
        (Optionally) RMSE map

    Example:
        >>> from torchmetrics.functional.image import root_mean_squared_error_using_sliding_window
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> root_mean_squared_error_using_sliding_window(preds, target)
        tensor(0.3999)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.
    """
    if not isinstance(window_size, int) or isinstance(window_size, int) and window_size < 1:
        raise ValueError("Argument `window_size` is expected to be a positive integer.")

    rmse_val_sum, rmse_map, total_images = _rmse_sw_update(
        preds, target, window_size, rmse_val_sum=None, rmse_map=None, total_images=None
    )
    rmse, rmse_map = _rmse_sw_compute(rmse_val_sum, rmse_map, total_images)

    if return_rmse_map:
        return rmse, rmse_map
    return rmse
