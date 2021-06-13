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


def si_sdr(target: Tensor, preds: Tensor, zero_mean: bool = False) -> Tensor:
    """ scale-invariant signal-to-distortion ratio (SI-SDR)

    Args:
        target:
            shape [..., time]
        preds:
            shape [..., time]
        zero_mean:
            if to zero mean target and preds or not

    Raises:
        TypeError:
            if target and preds have a different shape

    Returns:
        si-sdr value of shape [...]

    Example:
        >>> from torchmetrics.functional.audio import si_sdr
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_sdr_val = si_sdr(target, preds)
        >>> si_sdr_val
        tensor(18.4030)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
         and Signal Processing (ICASSP) 2019.
    """

    if target.shape != preds.shape:
        raise TypeError(f"Inputs must be of shape [..., time], got {target.shape} and {preds.shape} instead")

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    α = torch.sum(preds * target, dim=-1, keepdim=True) / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
    target_scaled = α * target

    noise = target_scaled - preds

    si_sdr_value = torch.sum(target_scaled**2, dim=-1) / (torch.sum(noise**2, dim=-1) + 1e-8)
    si_sdr_value = 10 * torch.log10(si_sdr_value + 1e-8)

    return si_sdr_value
