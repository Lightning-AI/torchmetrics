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
from typing_extensions import Literal

from torchmetrics.utilities.data import dim_zero_cat


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

    preds_vif = torch.zeros(preds.size(0), dtype=dtype, device=device)
    target_vif = torch.zeros(preds.size(0), dtype=dtype, device=device)

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

        preds_vif += torch.sum(torch.log10(1.0 + (g**2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)), dim=[1, 2, 3])
        target_vif += torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])

    return preds_vif / target_vif


def visual_information_fidelity(
    preds: Tensor,
    target: Tensor,
    sigma_n_sq: float = 2.0,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """Compute Pixel-Based Visual Information Fidelity (VIF-P).

    VIF is a full-reference metric that measures the amount of visual information
    preserved in a distorted image compared to the reference image.

    Args:
        preds: Predicted images of shape (N, C, H, W). Height and width must be at least 41.
        target: Ground truth images of shape (N, C, H, W). Must match preds in shape.
        sigma_n_sq: Variance of the visual noise. Default: 2.0.
        reduction: Method for reducing the metric across the batch.
            - "mean": Return a tensor average over the batch.
            - "none": Return a VIF score for each sample as a 1D tensor of shape (N,).

    Returns:
        torch.Tensor: VIF score(s). The shape depends on the `reduction` argument:
            - If ``reduction="mean"``, returns a scalar tensor.
            - If ``reduction="none"``, returns a tensor of shape ``(N,)``.

    Raises:
        ValueError: If input dimensions are smaller than ``41x41``.
        ValueError: If ``preds`` and ``target`` shapes don't match.
        ValueError: If ``reduction`` is not ``"mean"`` or ``"none"``.

    Example:
        >>> from torchmetrics.functional.image import visual_information_fidelity
        >>> preds = torch.randn(4, 3, 41, 41, generator=torch.Generator().manual_seed(42))
        >>> target = torch.randn(4, 3, 41, 41, generator=torch.Generator().manual_seed(43))
        >>> visual_information_fidelity(preds, target, reduction="none")
        tensor([0.0040, 0.0049, 0.0017, 0.0039])

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

    if preds.shape != target.shape:
        raise ValueError(f"`preds` and `target` must have the same shape, but got {preds.shape} vs {target.shape}.")

    if reduction not in ("mean", "none"):
        raise ValueError(f"Argument `reduction` must be 'mean' or 'none', but got {reduction}")

    per_channel_scores = [
        _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], sigma_n_sq) for i in range(preds.size(1))
    ]

    vif_per_sample = dim_zero_cat(
        torch.stack(per_channel_scores, dim=0).mean(0) if preds.size(1) > 1 else per_channel_scores[0]
    )

    if reduction == "mean":
        return vif_per_sample.mean()
    return vif_per_sample
