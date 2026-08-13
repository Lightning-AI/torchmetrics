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
from typing import Any

import numpy as np
import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _MULTIPROCESSING_AVAILABLE, _PESQ_AVAILABLE

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
    sharpness, call volume, background noise, clipping, audio interference etc. PESQ returns a score between -0.5 and
    4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to `cpu` to perform the metric
    calculation.

    .. hint::
        Usingsing this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``. Note that ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.

    .. note::
        Samples that the ``pesq`` backend cannot score, e.g. a silent reference for which no utterance can be
        detected, are returned as ``nan`` instead of raising an error, so that a single degenerate sample does not
        abort the calculation for the rest of the batch. The class based metric
        :class:`~torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality` excludes such samples from its average.

    Returns:
        Float tensor with shape ``(...,)`` of PESQ values per sample, with ``nan`` for samples the backend
        could not score

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
        >>> preds = randn(8000)
        >>> target = randn(8000)
        >>> perceptual_evaluation_speech_quality(preds, target, 8000, 'nb')
        tensor(2.2885)
        >>> perceptual_evaluation_speech_quality(preds, target, 16000, 'wb')
        tensor(1.6805)

    """
    if not _PESQ_AVAILABLE:
        raise ModuleNotFoundError(
            "PESQ metric requires that pesq is installed."
            " Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
        )
    import pesq as pesq_backend

    def _issubtype_number(x: Any) -> bool:
        return np.issubdtype(type(x), np.number)

    _filter_error_msg = np.vectorize(_issubtype_number, otypes=[bool])

    # with ``on_error=PesqError.RETURN_VALUES`` the backend reports a failure as one of these negative codes
    # instead of raising; all valid PESQ scores are >= -0.5 so the codes cannot collide with a real score
    error_codes = [
        code
        for name, code in vars(pesq_backend.PesqError).items()
        if not name.startswith("_") and isinstance(code, int) and code < 0
    ]

    def _errors_to_nan(values: Any) -> np.ndarray:
        """Convert raw backend outputs into float scores, mapping every failure onto ``nan``.

        A failure is reported either as an error code (``pesq``/``pesq_batch`` with
        ``on_error=PesqError.RETURN_VALUES``) or as an exception object (``pesq_batch`` collects exceptions
        raised inside its worker processes). Both are replaced by ``nan``, keeping one entry per input sample.

        """
        values = np.asarray(values, dtype=object).reshape(-1)
        scores = np.where(_filter_error_msg(values), values, np.nan).astype(np.float32)
        scores[np.isin(scores, error_codes)] = np.nan
        return scores

    if fs not in (8000, 16000):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
    if mode not in ("wb", "nb"):
        raise ValueError(f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        pesq_val_np = pesq_backend.pesq(
            fs,
            target.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            mode,
            on_error=pesq_backend.PesqError.RETURN_VALUES,
        )
        pesq_val = torch.tensor(_errors_to_nan(pesq_val_np)[0])
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()

        if _MULTIPROCESSING_AVAILABLE and n_processes != 1:
            pesq_val_np = pesq_backend.pesq_batch(
                fs,
                target_np,
                preds_np,
                mode,
                n_processor=n_processes,
                on_error=pesq_backend.PesqError.RETURN_VALUES,
            )
        else:
            pesq_val_np = [
                pesq_backend.pesq(
                    fs,
                    target_np[b, :],
                    preds_np[b, :],
                    mode,
                    on_error=pesq_backend.PesqError.RETURN_VALUES,
                )
                for b in range(preds_np.shape[0])
            ]
        pesq_val = torch.from_numpy(_errors_to_nan(pesq_val_np))
        pesq_val = pesq_val.reshape(preds.shape[:-1])

    if keep_same_device:
        return pesq_val.to(preds.device)

    return pesq_val
