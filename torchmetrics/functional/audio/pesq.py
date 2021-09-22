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
import pesq as pesq_backend
import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def pesq(preds: Tensor, target: Tensor, fs: int, mode: str, keep_same_device: bool = False) -> Tensor:
    r"""PESQ (Perceptual Evaluation of Speech Quality)

    This is a wrapper for the ``pesq`` package [1].

     .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install 
         torchmetrics[audio]`` or ``pip install pesq`` 

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        fs:
            sampling frequency, should be 16000 or 8000 (Hz)
        mode:
            'wb' (wide-band) or 'nb' (narrow-band)
        keep_same_device:
            whether to move the pesq value to the device of preds

    Returns:
        pesq value of shape [...]

    Example:
        >>> from torchmetrics.functional.audio import pesq
        >>> import torch
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> nb_pesq_val = pesq(preds, target, 8000, 'nb')
        >>> wb_pesq_val = pesq(preds, target, 16000, 'wb')

    References:
        [1] https://github.com/ludlows/python-pesq
    """
    if fs not in (8000, 16000):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
    if mode not in ("wb", "nb"):
        raise ValueError("Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        pesq_val_np = pesq_backend.pesq(fs, target.detach().cpu().numpy(), preds.detach().cpu().numpy(), mode)
        pesq_val = torch.tensor(pesq_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        pesq_val_np = np.empty(shape=(preds_np.shape[0]))
        for b in range(preds_np.shape[0]):
            pesq_val_np[b] = pesq_backend.pesq(fs, target_np[b, :], preds_np[b, :], mode)
        pesq_val = torch.from_numpy(pesq_val_np)
        pesq_val = pesq_val.reshape(preds.shape[:-1])

    if keep_same_device:
        pesq_val = pesq_val.to(preds.device)

    return pesq_val
