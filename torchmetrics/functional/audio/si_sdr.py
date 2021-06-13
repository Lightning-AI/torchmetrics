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


def si_sdr(target: Tensor, estimate: Tensor, zero_mean: bool = False, EPS: bool = 1e-8) -> Tensor:
    """ scale-invariant signal-to-distortion ratio (SI-SDR)

    Args:
        target (Tensor): shape [..., time]
        preds (Tensor): shape [..., time]
        zero_mean (Bool): if to zero mean target and estimate or not
        EPS (float, optional): a small value for numerical stability. Defaults to 1e-8.

    Raises:
        TypeError: if target and estimate have a different shape

    Returns:
        Tensor: si-sdr value has a shape of [...]

    Example:
        >>> from torchmetrics.functional.audio import si_sdr
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> estimate = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_sdr_val = si_sdr(target,estimate)
        >>> si_sdr_val
        tensor(18.4030)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
         and Signal Processing (ICASSP) 2019.
    """

    if target.shape != estimate.shape:
        raise TypeError(f"Inputs must be of shape [..., time], got {target.shape} and {estimate.shape} instead")

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    α = torch.sum(estimate * target, dim=-1, keepdim=True) / (torch.sum(target**2, dim=-1, keepdim=True) + EPS)
    target_scaled = α * target

    noise = target_scaled - estimate

    si_sdr_value = torch.sum(target_scaled**2, dim=-1) / (torch.sum(noise**2, dim=-1) + EPS)
    si_sdr_value = 10 * torch.log10(si_sdr_value + EPS)

    return si_sdr_value
