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

from typing import Dict, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce


def _spatial_distortion_index_update(preds: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Update and returns variables required to compute Spatial Distortion Index.

    Args:
        preds: High resolution multispectral image.
        target: A dictionary containing the following keys:

            - ``'ms'``: low resolution multispectral image.
            - ``'pan'``: high resolution panchromatic image.
            - ``'pan_lr'``: (optional) low resolution panchromatic image.

    Return:
        A tuple of Tensors containing ``preds`` and ``target``.

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds`` and ``target`` don't have the same batch and channel sizes.
        ValueError:
            If ``target`` doesn't have ``ms`` and ``pan``.

    """
    if len(preds.shape) != 4:
        raise ValueError(f"Expected `preds` to have BxCxHxW shape. Got preds: {preds.shape}.")
    if "ms" not in target or "pan" not in target:
        raise ValueError(f"Expected `target` to have keys ('ms', 'pan'). Got target: {target.keys()}")
    for name, t in target.items():
        if preds.dtype != t.dtype:
            raise TypeError(
                f"Expected `preds` and `{name}` to have the same data type. "
                "Got preds: {preds.dtype} and {name}: {t.dtype}."
            )
    for name, t in target.items():
        if len(t.shape) != 4:
            raise ValueError(f"Expected `{name}` to have BxCxHxW shape. Got {name}: {t.shape}.")
    for name, t in target.items():
        if preds.shape[:2] != t.shape[:2]:
            raise ValueError(
                f"Expected `preds` and `{name}` to have same batch and channel sizes. "
                "Got preds: {preds.shape} and {name}: {t.shape}."
            )
    return preds, target


def _spatial_distortion_index_compute(
    preds: Tensor,
    target: Dict[str, Tensor],
    p: int = 1,
    ws: int = 7,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Compute Spatial Distortion Index (SpatialDistortionIndex_).

    Args:
        preds: High resolution multispectral image.
        target: A dictionary containing the following keys:

            - ``'ms'``: low resolution multispectral image.
            - ``'pan'``: high resolution panchromatic image.
            - ``'pan_lr'``: (optional) low resolution panchromatic image.

        p: Order of the norm applied on the difference.
        ws: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.

    Example:
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> target = {
        ...     'ms': torch.rand([16, 3, 16, 16]),
        ...     'pan': torch.rand([16, 3, 32, 32]),
        ... }
        >>> preds, target = _spatial_distortion_index_update(preds, target)
        >>> _spatial_distortion_index_compute(preds, target)
        tensor(0.0090)

    """
    length = preds.shape[1]

    ms = target["ms"]
    pan = target["pan"]
    pan_lr = target["pan_lr"] if "pan_lr" in target else None

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
    if ws >= ms_h or ws >= ms_w:
        raise ValueError(f"Expected `ws` to be smaller than dimension of `ms`. Got ws: {ws}.")

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

    if pan_lr is None:
        from torchvision.transforms.functional import resize

        from torchmetrics.functional.image.helper import _uniform_filter

        pan_degraded = _uniform_filter(pan, window_size=ws)
        pan_degraded = resize(pan_degraded, size=ms.shape[-2:], antialias=False)
    else:
        pan_degraded = pan_lr

    m1 = torch.zeros(length, device=preds.device)
    m2 = torch.zeros(length, device=preds.device)

    for i in range(length):
        m1[i] = universal_image_quality_index(ms[:, i : i + 1], pan_degraded[:, i : i + 1])
        m2[i] = universal_image_quality_index(preds[:, i : i + 1], pan[:, i : i + 1])
    diff = (m1 - m2).abs() ** p
    return reduce(diff, reduction) ** (1 / p)


def spatial_distortion_index(
    preds: Tensor,
    target: Dict[str, Tensor],
    p: int = 1,
    ws: int = 7,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Calculate `Spatial Distortion Index`_ (SpatialDistortionIndex_) also known as D_s.

    Metric is used to compare the spatial distortion between two images.

    Args:
        preds: High resolution multispectral image.
        target: A dictionary containing the following keys:

            - ``'ms'``: low resolution multispectral image.
            - ``'pan'``: high resolution panchromatic image.
            - ``'pan_lr'``: (optional) low resolution panchromatic image.

        p: Order of the norm applied on the difference.
        ws: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds`` and ``target`` don't have the same batch and channel sizes.
        ValueError:
            If ``target`` doesn't have ``ms`` and ``pan``.
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.
        ValueError:
            If ``p`` is not a positive integer.
        ValueError:
            If ``ws`` is not a positive integer.

    Example:
        >>> from torchmetrics.functional.image import spatial_distortion_index
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> target = {
        ...     'ms': torch.rand([16, 3, 16, 16]),
        ...     'pan': torch.rand([16, 3, 32, 32]),
        ... }
        >>> spatial_distortion_index(preds, target)
        tensor(0.0090)

    """
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
    if not isinstance(ws, int) or ws <= 0:
        raise ValueError(f"Expected `ws` to be a positive integer. Got ws: {ws}.")
    preds, target = _spatial_distortion_index_update(preds, target)
    return _spatial_distortion_index_compute(preds, target, p, ws, reduction)
