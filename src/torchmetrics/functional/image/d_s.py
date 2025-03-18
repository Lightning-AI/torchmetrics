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

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["_spatial_distortion_index_compute", "spatial_distortion_index"]


def _spatial_distortion_index_update(
    preds: Tensor, ms: Tensor, pan: Tensor, pan_lr: Optional[Tensor] = None
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Update and returns variables required to compute Spatial Distortion Index.

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.

    Return:
        A tuple of Tensors containing ``preds``, ``ms``, ``pan`` and ``pan_lr``.

    Raises:
        TypeError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same data type.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same batch and channel sizes.
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.

    """
    if len(preds.shape) != 4:
        raise ValueError(f"Expected `preds` to have BxCxHxW shape. Got preds: {preds.shape}.")
    if preds.dtype != ms.dtype:
        raise TypeError(
            f"Expected `preds` and `ms` to have the same data type. Got preds: {preds.dtype} and ms: {ms.dtype}."
        )
    if preds.dtype != pan.dtype:
        raise TypeError(
            f"Expected `preds` and `pan` to have the same data type. Got preds: {preds.dtype} and pan: {pan.dtype}."
        )
    if pan_lr is not None and preds.dtype != pan_lr.dtype:
        raise TypeError(
            f"Expected `preds` and `pan_lr` to have the same data type."
            f" Got preds: {preds.dtype} and pan_lr: {pan_lr.dtype}."
        )
    if len(ms.shape) != 4:
        raise ValueError(f"Expected `ms` to have BxCxHxW shape. Got ms: {ms.shape}.")
    if len(pan.shape) != 4:
        raise ValueError(f"Expected `pan` to have BxCxHxW shape. Got pan: {pan.shape}.")
    if pan_lr is not None and len(pan_lr.shape) != 4:
        raise ValueError(f"Expected `pan_lr` to have BxCxHxW shape. Got pan_lr: {pan_lr.shape}.")
    if preds.shape[:2] != ms.shape[:2]:
        raise ValueError(
            f"Expected `preds` and `ms` to have the same batch and channel sizes."
            f" Got preds: {preds.shape} and ms: {ms.shape}."
        )
    if preds.shape[:2] != pan.shape[:2]:
        raise ValueError(
            f"Expected `preds` and `pan` to have the same batch and channel sizes."
            f" Got preds: {preds.shape} and pan: {pan.shape}."
        )
    if pan_lr is not None and preds.shape[:2] != pan_lr.shape[:2]:
        raise ValueError(
            f"Expected `preds` and `pan_lr` to have the same batch and channel sizes."
            f" Got preds: {preds.shape} and pan_lr: {pan_lr.shape}."
        )

    preds_h, preds_w = preds.shape[-2:]
    ms_h, ms_w = ms.shape[-2:]
    pan_h, pan_w = pan.shape[-2:]
    if preds_h != pan_h:
        raise ValueError(f"Expected `preds` and `pan` to have the same height. Got preds: {preds_h} and pan: {pan_h}")
    if preds_w != pan_w:
        raise ValueError(f"Expected `preds` and `pan` to have the same width. Got preds: {preds_w} and pan: {pan_w}")
    if preds_h % ms_h != 0:
        raise ValueError(
            f"Expected height of `preds` to be multiple of height of `ms`. Got preds: {preds_h} and ms: {ms_h}."
        )
    if preds_w % ms_w != 0:
        raise ValueError(
            f"Expected width of `preds` to be multiple of width of `ms`. Got preds: {preds_w} and ms: {ms_w}."
        )
    if pan_h % ms_h != 0:
        raise ValueError(
            f"Expected height of `pan` to be multiple of height of `ms`. Got preds: {pan_h} and ms: {ms_h}."
        )
    if pan_w % ms_w != 0:
        raise ValueError(f"Expected width of `pan` to be multiple of width of `ms`. Got preds: {pan_w} and ms: {ms_w}.")

    if pan_lr is not None:
        pan_lr_h, pan_lr_w = pan_lr.shape[-2:]
        if pan_lr_h != ms_h:
            raise ValueError(
                f"Expected `ms` and `pan_lr` to have the same height. Got ms: {ms_h} and pan_lr: {pan_lr_h}."
            )
        if pan_lr_w != ms_w:
            raise ValueError(
                f"Expected `ms` and `pan_lr` to have the same width. Got ms: {ms_w} and pan_lr: {pan_lr_w}."
            )

    return preds, ms, pan, pan_lr


def _spatial_distortion_index_compute(
    preds: Tensor,
    ms: Tensor,
    pan: Tensor,
    pan_lr: Optional[Tensor] = None,
    norm_order: int = 1,
    window_size: int = 7,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Compute Spatial Distortion Index (SpatialDistortionIndex_).

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        ValueError
            If ``window_size`` is smaller than dimension of ``ms``.

    Example:
        >>> from torch import rand
        >>> preds = rand([16, 3, 32, 32])
        >>> ms = rand([16, 3, 16, 16])
        >>> pan = rand([16, 3, 32, 32])
        >>> preds, ms, pan, pan_lr = _spatial_distortion_index_update(preds, ms, pan)
        >>> _spatial_distortion_index_compute(preds, ms, pan, pan_lr)
        tensor(0.0090)

    """
    length = preds.shape[1]

    ms_h, ms_w = ms.shape[-2:]
    if window_size >= ms_h or window_size >= ms_w:
        raise ValueError(
            f"Expected `window_size` to be smaller than dimension of `ms`. Got window_size: {window_size}."
        )

    if pan_lr is None:
        if not _TORCHVISION_AVAILABLE:
            raise ValueError(
                "When `pan_lr` is not provided as input to metric Spatial distortion index, torchvision should be "
                "installed. Please install with `pip install torchvision` or `pip install torchmetrics[image]`."
            )
        from torchvision.transforms.functional import resize

        from torchmetrics.functional.image.utils import _uniform_filter

        pan_degraded = _uniform_filter(pan, window_size=window_size)
        pan_degraded = resize(pan_degraded, size=ms.shape[-2:], antialias=False)
    else:
        pan_degraded = pan_lr

    m1 = torch.zeros(length, device=preds.device)
    m2 = torch.zeros(length, device=preds.device)

    for i in range(length):
        m1[i] = universal_image_quality_index(ms[:, i : i + 1], pan_degraded[:, i : i + 1])
        m2[i] = universal_image_quality_index(preds[:, i : i + 1], pan[:, i : i + 1])
    diff = (m1 - m2).abs() ** norm_order
    return reduce(diff, reduction) ** (1 / norm_order)


def spatial_distortion_index(
    preds: Tensor,
    ms: Tensor,
    pan: Tensor,
    pan_lr: Optional[Tensor] = None,
    norm_order: int = 1,
    window_size: int = 7,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Calculate `Spatial Distortion Index`_ (SpatialDistortionIndex_) also known as D_s.

    Metric is used to compare the spatial distortion between two images.

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        TypeError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same data type.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same batch and channel sizes.
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.
        ValueError:
            If ``norm_order`` is not a positive integer.
        ValueError:
            If ``window_size`` is not a positive integer.

    Example:
        >>> from torch import rand
        >>> from torchmetrics.functional.image import spatial_distortion_index
        >>> preds = rand([16, 3, 32, 32])
        >>> ms = rand([16, 3, 16, 16])
        >>> pan = rand([16, 3, 32, 32])
        >>> spatial_distortion_index(preds, ms, pan)
        tensor(0.0090)

    """
    if not isinstance(norm_order, int) or norm_order <= 0:
        raise ValueError(f"Expected `norm_order` to be a positive integer. Got norm_order: {norm_order}.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"Expected `window_size` to be a positive integer. Got window_size: {window_size}.")
    preds, ms, pan, pan_lr = _spatial_distortion_index_update(preds, ms, pan, pan_lr)
    return _spatial_distortion_index_compute(preds, ms, pan, pan_lr, norm_order, window_size, reduction)
