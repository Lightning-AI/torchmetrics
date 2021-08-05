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
import numpy as np
from mir_eval.separation import bss_eval_sources
import torch
from torch import Tensor
import warnings
from torchmetrics.utilities.checks import _check_same_shape


def sdr_sir_sar(preds: Tensor, target: Tensor, compute_permutation=False, keep_same_device: bool = False) -> Tensor:
    r"""sdr_sir_sar evaluates the SDR, SIR, SAR metrics of preds and target. sdr_sir_sar is a wrapper for the mir_eval.separation.bss_eval_sources function.

    Args:
        preds:
            shape ``[..., spk, time]``
        target:
            shape ``[..., spk, time]``
        compute_permutation:
            whether to compute the metrics permutation invariantly. By default, it is False for we can use PIT to compute the permutation in a better way and in the sense of any metrics.
        keep_same_device:
            whether to move the stoi value to the device of preds

    Returns:
        sdr value of shape [..., spk]
        sir value of shape [..., spk]
        sar value of shape [..., spk]

    Example:
        >>> from torchmetrics.functional.audio import sdr_sir_sar
        >>> import torch
        >>> preds = torch.randn(2, 8000)
        >>> target = torch.randn(2, 8000)
        >>> sdr_val, sir_val, sar_val = sdr_sir_sar(preds, target)

    """
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        warnings.warn('preds and target should be of the shape [..., spk, time], but 1d tensor detected')
        sdr_val_np, sir_val_np, sar_val_np, perm = bss_eval_sources(target.detach().cpu().numpy()[None, ...], preds.detach().cpu().numpy()[None, ...], compute_permutation)
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
            sdr_val_np[b,:], sir_val_np[b,:], sar_val_np[b,:], perm = bss_eval_sources(target_np[b,], preds_np[b,], compute_permutation)
        sdr_val = torch.tensor(sdr_val_np)
        sir_val = torch.tensor(sir_val_np)
        sar_val = torch.tensor(sar_val_np)

    if keep_same_device:
        sdr_val = sdr_val.to(preds.device)
        sir_val = sir_val.to(preds.device)
        sar_val = sar_val.to(preds.device)

    return sdr_val, sir_val, sar_val
