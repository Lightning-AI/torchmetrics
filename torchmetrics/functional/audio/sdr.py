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
from typing import Optional

import numpy as np
import torch

from torchmetrics.utilities.imports import _FAST_BSS_EVAL_AVAILABLE, _TORCH_GREATER_EQUAL_1_8

if _FAST_BSS_EVAL_AVAILABLE:
    if _TORCH_GREATER_EQUAL_1_8:
        from fast_bss_eval.torch.cgd import toeplitz_conjugate_gradient
        from fast_bss_eval.torch.helpers import _normalize
        from fast_bss_eval.torch.linalg import toeplitz
        from fast_bss_eval.torch.metrics import compute_stats
    else:
        from fast_bss_eval.numpy.cgd import toeplitz_conjugate_gradient
        from fast_bss_eval.numpy.helpers import _normalize
        from fast_bss_eval.numpy.linalg import toeplitz
        from fast_bss_eval.numpy.metrics import compute_stats
else:
    toeplitz = None
    toeplitz_conjugate_gradient = None
    compute_stats = None
    _normalize = None

from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def sdr(
    preds: Tensor,
    target: Tensor,
    use_cg_iter: Optional[int] = None,
    filter_length: int = 512,
    zero_mean: bool = False,
    load_diag: Optional[float] = None,
) -> Tensor:
    r"""Signal to Distortion Ratio (SDR) [1,2,3]

    .. note:: 1) using this metrics requires you to have ``fast-bss-eval`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install fast-bss-eval``
        2) preds and target need to have the same dtype, otherwise target will be converted to preds' dtype
        3) when pytorch<1.8.0, numpy will be used to calculate this metric, which causes ``sdr`` non-differentiable

    Args:
        preds:
            shape ``[..., time]``
        target:
            shape ``[..., time]``
        use_cg_iter:
            If provided, an iterative method is used to solve for the distortion \
            filter coefficients instead of direct Gaussian elimination. \
            This can speed up the computation of the metrics in case the filters \
            are long. Using a value of 10 here has been shown to provide \
            good accuracy in most cases and is sufficient when using this \
            loss to train neural separation networks.
        filter_length:
            The length of the distortion filter allowed
        zero_mean:
            When set to True, the mean of all signals is subtracted prior to computation of the metrics
        load_diag:
            If provided, this small value is added to the diagonal coefficients of \
            the system metrics when solving for the filter coefficients. \
            This can help stabilize the metric in the case where some of the reference \
            signals may sometimes be zero

    Raises:
        ValueError:
            If ``fast-bss-eval`` package is not installed

    Returns:
        sdr value of shape ``[...]`` if compute_permutation is False, else ``[..., spk]``

    Example:
        >>> from torchmetrics.functional.audio import sdr
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> sdr(preds, target)
        tensor(-12.0589)
        >>> # use with pit
        >>> from torchmetrics.functional.audio import pit
        >>> # [batch, spk, time]
        >>> preds = torch.randn(4, 2, 8000)
        >>> target = torch.randn(4, 2, 8000)
        >>> best_metric, best_perm = pit(preds, target, sdr, 'max')
        >>> best_metric
        tensor([-11.6375, -11.4358, -11.7148, -11.6325])
        >>> best_perm
        tensor([[1, 0],
                [0, 1],
                [1, 0],
                [0, 1]])

    References:
        [1] Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
         IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462–1469.
        [2] Scheibler, R. (2021). SDR -- Medium Rare with Fast Computations.
        [3] https://github.com/fakufaku/fast_bss_eval
    """

    if not _FAST_BSS_EVAL_AVAILABLE:
        raise ValueError(
            "SDR metric requires that fast-bss-eval is installed." \
            "Either install as `pip install torchmetrics[audio]` or `pip install fast-bss-eval`"
        )
    _check_same_shape(preds, target)

    if not preds.dtype.is_floating_point:
        preds = preds.float()  # for torch.norm

    if preds.dtype != target.dtype:  # for torch.linalg.solve
        target = target.to(preds.dtype)

    if zero_mean:
        preds = preds - preds.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    # use numpy if torch<1.8
    if not _TORCH_GREATER_EQUAL_1_8:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

    # normalize along time-axis
    preds = _normalize(preds, dim=-1)
    target = _normalize(target, dim=-1)

    # solve for the optimal filter
    # compute auto-correlation and cross-correlation
    acf, xcorr = compute_stats(target, preds, length=filter_length, pairwise=False)

    if load_diag is not None:
        # the diagonal factor of the Toeplitz matrix is the first
        # coefficient of the acf
        acf[..., 0] += load_diag

    if use_cg_iter is not None:
        # use preconditioned conjugate gradient
        sol = toeplitz_conjugate_gradient(acf, xcorr, n_iter=use_cg_iter)
    else:
        # regular matrix solver
        R_mat = toeplitz(acf)
        sol = torch.linalg.solve(R_mat, xcorr)

    # to tensor if torch<1.8
    if not _TORCH_GREATER_EQUAL_1_8:
        sol = torch.tensor(sol, device=preds.device)

    # compute the coherence
    coh = torch.einsum("...l,...l->...", xcorr, sol)

    # transform to decibels
    ratio = coh / (1 - coh)
    sdr_val = 10.0 * torch.log10(ratio)
    return sdr_val
