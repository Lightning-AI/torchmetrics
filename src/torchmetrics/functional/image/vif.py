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
from torch.nn.functional import conv2d

from torchmetrics.utilities.distributed import reduce


def _filter(win_size: float, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    # This code is inspired by
    # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/utils.py#L45
    # https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/functional/filters.py#L38
    # Both links do the same, but the second one is cleaner
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
    g = coords**2
    g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma**2))
    g /= torch.sum(g)
    return g


def _vif_per_channel(preds: Tensor, target: Tensor, sigma_n_sq: float) -> Tensor:
    dtype = preds.dtype
    device = preds.device

    preds = preds.unsqueeze(1)  # Add channel dimension
    target = target.unsqueeze(1)
    # Constant for numerical stability
    eps = torch.tensor(1e-10, dtype=dtype, device=device)

    sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)

    preds_vif, target_vif = torch.zeros(1, dtype=dtype, device=device), torch.zeros(1, dtype=dtype, device=device)
    for scale in range(4):
        n = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]

        if scale > 0:
            target = conv2d(target, kernel)[:, :, ::2, ::2]
            preds = conv2d(preds, kernel)[:, :, ::2, ::2]

        mu_target = conv2d(target, kernel)
        mu_preds = conv2d(preds, kernel)
        mu_target_sq = mu_target**2
        mu_preds_sq = mu_preds**2
        mu_target_preds = mu_target * mu_preds

        sigma_target_sq = torch.clamp(conv2d(target**2, kernel) - mu_target_sq, min=0.0)
        sigma_preds_sq = torch.clamp(conv2d(preds**2, kernel) - mu_preds_sq, min=0.0)
        sigma_target_preds = conv2d(target * preds, kernel) - mu_target_preds

        g = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_target_preds

        mask = sigma_target_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        sigma_target_sq[mask] = 0

        mask = sigma_preds_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = 0

        mask = g < 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        g[mask] = 0
        sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)

        preds_vif_scale = torch.log10(1.0 + (g**2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq))
        preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif = target_vif + torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])
    return preds_vif / target_vif


def visual_information_fidelity(preds: Tensor, target: Tensor, sigma_n_sq: float = 2.0) -> Tensor:
    """Compute Pixel Based Visual Information Fidelity (VIF_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``.
        target: ground truth images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``
        sigma_n_sq: variance of the visual noise

    Return:
        Tensor with vif-p score

    Raises:
        ValueError:
            If ``data_range`` is neither a ``tuple`` nor a ``float``

    """
    # This code is inspired by
    # https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/vif.py and
    # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/full_ref.py#L357.

    if preds.size(-1) < 41 or preds.size(-2) < 41:
        raise ValueError(f"Invalid size of preds. Expected at least 41x41, but got {preds.size(-1)}x{preds.size(-2)}!")

    if target.size(-1) < 41 or target.size(-2) < 41:
        raise ValueError(
            f"Invalid size of target. Expected at least 41x41, but got {target.size(-1)}x{target.size(-2)}!"
        )

    per_channel = [_vif_per_channel(preds[:, i, :, :], target[:, i, :, :], sigma_n_sq) for i in range(preds.size(1))]
    return reduce(torch.cat(per_channel), "elementwise_mean")
