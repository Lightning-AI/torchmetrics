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

import torch
from torch import Tensor

from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.checks import _check_same_shape


def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    r"""Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from torchmetrics.functional.audio import signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)
    """
    _check_same_shape(preds, target)
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    noise = target - preds

    snr_value = (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(snr_value)


def scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """`Scale-invariant signal-to-noise ratio`_ (SI-SNR).

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``

    Returns:
         Float tensor with shape ``(...,)`` of SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_noise_ratio(preds, target)
        tensor(15.0918)
    """
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=True)


def complex_scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """`Complex scale-invariant signal-to-noise ratio`_ (C-SI-SNR).

    Args:
        preds: real/complex float tensor with shape ``(..., frequency, time, 2)``/``(..., frequency, time)``
        target: real/complex float tensor with shape ``(..., frequency, time, 2)``/``(..., frequency, time)``
        zero_mean: When set to True, the mean of all signals is subtracted prior to computation of the metrics

    Returns:
         Float tensor with shape ``(...,)`` of C-SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` is not the shape (..., frequency, time, 2) (after being converted to real if it is complex).
            If ``preds`` and ``target`` does not have the same shape.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import complex_scale_invariant_signal_noise_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn((1,257,100,2))
        >>> target = torch.randn((1,257,100,2))
        >>> complex_scale_invariant_signal_noise_ratio(preds, target)
        tensor([-63.4849])
    """
    if preds.is_complex():
        preds = torch.view_as_real(preds)
    if target.is_complex():
        target = torch.view_as_real(target)

    if (preds.ndim < 3 or preds.shape[-1] != 2) or (target.ndim < 3 or target.shape[-1] != 2):
        raise RuntimeError(
            "Predictions and targets are expected to have the shape (..., frequency, time, 2),"
            " but got {preds.shape} and {target.shape}."
        )

    preds = preds.reshape(*preds.shape[:-3], -1)
    target = target.reshape(*target.shape[:-3], -1)

    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=zero_mean)
