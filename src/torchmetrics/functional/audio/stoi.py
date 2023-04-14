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
import numpy as np
import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _PYSTOI_AVAILABLE

if _PYSTOI_AVAILABLE:
    from pystoi import stoi as stoi_backend
else:
    stoi_backend = None
    __doctest_skip__ = ["short_time_objective_intelligibility"]


def short_time_objective_intelligibility(
    preds: Tensor, target: Tensor, fs: int, extended: bool = False, keep_same_device: bool = False
) -> Tensor:
    r"""Calculate STOI (Short-Time Objective Intelligibility) metric for evaluating speech signals.

    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due to
    additive noise, single-/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations. The
    STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good alternative
    to the speech intelligibility index (SII) or the speech transmission index (STI), when you are interested in
    the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms, on speech
    intelligibility. Description taken from  `Cees Taal's website`_ and for further defails see `STOI ref1`_ and
    `STOI ref2`_.

    This metric is a wrapper for the `pystoi package`_. As the implementation backend implementation only supports
    calculations on CPU, all input will automatically be moved to CPU to perform the metric calculation before being
    moved back to the original device.

    .. note:: using this metrics requires you to have ``pystoi`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pystoi``

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        fs: sampling frequency (Hz)
        extended: whether to use the extended STOI described in `STOI ref3`_.
        keep_same_device: whether to move the stoi value to the device of preds

    Returns:
        stoi value of shape [...]

    Raises:
        ModuleNotFoundError:
            If ``pystoi`` package is not installed
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> short_time_objective_intelligibility(preds, target, 8000).float()
        tensor(-0.0100)
    """
    if not _PYSTOI_AVAILABLE:
        raise ModuleNotFoundError(
            "ShortTimeObjectiveIntelligibility metric requires that `pystoi` is installed."
            " Either install as `pip install torchmetrics[audio]` or `pip install pystoi`."
        )
    _check_same_shape(preds, target)

    if len(preds.shape) == 1:
        stoi_val_np = stoi_backend(target.detach().cpu().numpy(), preds.detach().cpu().numpy(), fs, extended)
        stoi_val = torch.tensor(stoi_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        stoi_val_np = np.empty(shape=(preds_np.shape[0]))
        for b in range(preds_np.shape[0]):
            stoi_val_np[b] = stoi_backend(target_np[b, :], preds_np[b, :], fs, extended)
        stoi_val = torch.from_numpy(stoi_val_np)
        stoi_val = stoi_val.reshape(preds.shape[:-1])

    if keep_same_device:
        return stoi_val.to(preds.device)

    return stoi_val
