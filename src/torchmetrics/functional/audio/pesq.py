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
from torchmetrics.utilities.imports import _MULTIPROCESSING_AVAILABLE, _PESQ_AVAILABLE

if _PESQ_AVAILABLE:
    import pesq as pesq_backend
else:
    pesq_backend = None


__doctest_requires__ = {("perceptual_evaluation_speech_quality",): ["pesq"]}


def perceptual_evaluation_speech_quality(
    preds: Tensor,
    target: Tensor,
    fs: int,
    mode: str,
    keep_same_device: bool = False,
    n_processes: int = 1,
) -> Tensor:
    r"""Calculate `Perceptual Evaluation of Speech Quality`_ (PESQ).

    It's a recognized industry standard for audio quality that takes into considerations characteristics such as: audio
    sharpness, call volume, background noise, clipping, audio interference ect. PESQ returns a score between -0.5 and
    4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to `cpu` to perform the metric
    calculation.

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``. Note that ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifiying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.

    Returns:
        Float tensor with shape ``(...,)`` of PESQ values per sample

    Raises:
        ModuleNotFoundError:
            If ``pesq`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``
        RuntimeError:
            If ``preds`` and ``target`` do not have the same shape

    Example:
        >>> from torch import randn
        >>> from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
        >>> g = torch.manual_seed(1)
        >>> preds = randn(8000)
        >>> target = randn(8000)
        >>> perceptual_evaluation_speech_quality(preds, target, 8000, 'nb')
        tensor(2.2076)
        >>> perceptual_evaluation_speech_quality(preds, target, 16000, 'wb')
        tensor(1.7359)
    """
    if not _PESQ_AVAILABLE:
        raise ModuleNotFoundError(
            "PESQ metric requires that pesq is installed."
            " Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
        )
    if fs not in (8000, 16000):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
    if mode not in ("wb", "nb"):
        raise ValueError(f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        pesq_val_np = pesq_backend.pesq(fs, target.detach().cpu().numpy(), preds.detach().cpu().numpy(), mode)
        pesq_val = torch.tensor(pesq_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()

        if _MULTIPROCESSING_AVAILABLE and n_processes != 1:
            pesq_val_np = pesq_backend.pesq_batch(fs, target_np, preds_np, mode, n_processor=n_processes)
            pesq_val_np = np.array(pesq_val_np)
        else:
            pesq_val_np = np.empty(shape=(preds_np.shape[0]))
            for b in range(preds_np.shape[0]):
                pesq_val_np[b] = pesq_backend.pesq(fs, target_np[b, :], preds_np[b, :], mode)
        pesq_val = torch.from_numpy(pesq_val_np)
        pesq_val = pesq_val.reshape(preds.shape[:-1])

    if keep_same_device:
        return pesq_val.to(preds.device)

    return pesq_val
