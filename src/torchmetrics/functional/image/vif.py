"""
This code is inspired by
https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/vif.py and
https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/full_ref.py#L357

Reference: https://ieeexplore.ieee.org/abstract/document/1576816
https://live.ece.utexas.edu/research/quality/VIF.htm
"""
from typing import Literal
from typing import Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import Tensor

from torchmetrics.functional.image.helper import _gaussian_kernel_2d
from torchmetrics.utilities.distributed import reduce


def visual_information_fidelity(
        preds: Tensor,
        target: Tensor,
        data_range: Optional[Tuple[float, float]],
        sigma_n_sq: float = 2.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean"
) -> Tensor:
    if data_range is not None:
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])

    # Constant for numerical stability
    eps = torch.tensor(1e-10)

    channel = preds.size(1)
    sigma_n_sq = torch.tensor(sigma_n_sq)

    preds_vif, target_vif = torch.zeros(1, dtype=preds.dtype), torch.zeros(1, dtype=preds.dtype)
    for scale in range(1, 5):
        kernel_size = 2 ** (4 - scale + 1) + 1
        kernel_size = (kernel_size, kernel_size)
        sigma = (kernel_size[0] / 5, kernel_size[1] / 5)
        kernel = _gaussian_kernel_2d(channel=channel, kernel_size=kernel_size, sigma=sigma, dtype=preds.dtype,
                                     device=preds.device)

        if scale > 1:
            preds = functional.conv2d(preds ** 2, kernel)[::2, ::2]
            target = functional.conv2d(target ** 2, kernel)[::2, ::2]

        mu_preds = functional.conv2d(preds, kernel)
        mu_target = functional.conv2d(target, kernel)
        mu_preds_sq = mu_preds ** 2
        mu_target_sq = mu_target ** 2
        mu_preds_target = mu_preds * mu_target

        sigma_preds_sq = torch.clamp(functional.conv2d(preds ** 2, kernel) - mu_preds_sq, min=0.0)
        sigma_target_sq = torch.clamp(functional.conv2d(target ** 2, kernel) - mu_target_sq, min=0.0)
        sigma_preds_target = functional.conv2d(preds * target, kernel) - mu_preds_target

        g = sigma_preds_target / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_preds_target

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

        preds_vif_scale = torch.log10(1.0 + (g ** 2.) * sigma_target_sq / (sigma_v_sq + sigma_n_sq))
        preds_vif += torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif += torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])
    score: Tensor = (preds_vif + eps) / (target_vif + eps)
    return reduce(x=score, reduction=reduction)
