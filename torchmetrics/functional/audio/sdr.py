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
import warnings
from typing import Tuple

import numpy as np
import torch

from torchmetrics.utilities.imports import _MIR_EVAL_AVAILABLE

if _MIR_EVAL_AVAILABLE:
    from mir_eval.separation import bss_eval_sources
else:
    bss_eval_sources = None

from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _sdr_sir_sar(
    preds: Tensor, target: Tensor, compute_permutation: bool = False, keep_same_device: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""_sdr_sir_sar evaluates the SDR, SIR, SAR metrics of preds and target. _sdr_sir_sar is a wrapper for the 
    mir_eval.separation.bss_eval_sources function.

    Args:
        preds:
            shape ``[..., time]`` if compute_permutation is False, else ``[..., spk, time]``
        target:
            shape ``[..., time]`` if compute_permutation is False, else ``[..., spk, time]``
        compute_permutation:
            whether to compute the metrics permutation invariantly. By default, it is False
            for we can use PIT to compute the permutation in a better way and in the sense of any metrics.
        keep_same_device:
            whether to move the metric value to the device of preds

    Raises:
        ValueError:
            If ``mir_eval`` package is not installed, or 1D input is given when compute_permutation is True

    Returns:
        sdr value of shape ``[...]`` if compute_permutation is False, else ``[..., spk]``
        sir value of shape ``[...]`` if compute_permutation is False, else ``[..., spk]``
        sar value of shape ``[...]`` if compute_permutation is False, else ``[..., spk]``

    """
    if not _MIR_EVAL_AVAILABLE:
        raise ValueError(
            "SDR metric requires that mir_eval is installed."
            "Either install as `pip install torchmetrics[audio]` or `pip install mir_eval`"
        )

    _check_same_shape(preds, target)

    if preds.ndim == 1:
        if compute_permutation:
            raise ValueError(
                "SDR metric requires preds and target to be of shape [..., spk, time]"
                " if compute_permutation is True, but 1D Tensor is given."
            )

        sdr_val_np, sir_val_np, sar_val_np, perm = bss_eval_sources(
            target.detach().cpu().numpy()[None, ...],
            preds.detach().cpu().numpy()[None, ...],
            compute_permutation,
        )
        sdr_val = torch.tensor(sdr_val_np[0])
        sir_val = torch.tensor(sir_val_np[0])
        sar_val = torch.tensor(sar_val_np[0])
    elif preds.ndim == 2:
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        sdr_val_np, sir_val_np, sar_val_np, perm = bss_eval_sources(target_np, preds_np, compute_permutation)
        sdr_val = torch.tensor(sdr_val_np)
        sir_val = torch.tensor(sir_val_np)
        sar_val = torch.tensor(sar_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-2], preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-2], preds.shape[-1]).detach().cpu().numpy()
        sdr_val_np = np.empty(preds_np.shape[:-1])
        sir_val_np = np.empty(preds_np.shape[:-1])
        sar_val_np = np.empty(preds_np.shape[:-1])
        for b in range(preds_np.shape[0]):
            sdr_val_np[b, :], sir_val_np[b, :], sar_val_np[b, :], perm = bss_eval_sources(
                target_np[
                    b,
                ],
                preds_np[
                    b,
                ],
                compute_permutation,
            )
        sdr_val = torch.tensor(sdr_val_np)
        sir_val = torch.tensor(sir_val_np)
        sar_val = torch.tensor(sar_val_np)

    if keep_same_device:
        sdr_val = sdr_val.to(preds.device)
        sir_val = sir_val.to(preds.device)
        sar_val = sar_val.to(preds.device)

    return sdr_val, sir_val, sar_val


def sdr(preds: Tensor, target: Tensor, compute_permutation: bool = False, keep_same_device: bool = False) -> Tensor:
    r"""sdr evaluates the Signal to Distortion Ratio (SDR) metric of preds and target. sdr is a wrapper for the 
    mir_eval.separation.bss_eval_sources function.

    Args:
        preds:
            shape ``[..., time]`` if compute_permutation is False, else ``[..., spk, time]``
        target:
            shape ``[..., time]`` if compute_permutation is False, else ``[..., spk, time]``
        compute_permutation:
            whether to compute the metrics permutation invariantly. By default, it is False
            for we can use PIT to compute the permutation in a better way and in the sense of any metrics.
        keep_same_device:
            whether to move the metric value to the device of preds

    Raises:
        ValueError:
            If ``mir_eval`` package is not installed, or 1D input is given when compute_permutation is True

    Returns:
        sdr value of shape ``[...]`` if compute_permutation is False, else ``[..., spk]``

    Example:
        >>> from torchmetrics.functional.audio import sdr
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> sdr(preds, target).float()
        tensor(-12.0589)
    """

    return _sdr_sir_sar(preds, target, compute_permutation, keep_same_device)[0]
