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

# Note: without special mention, the functions in this file are mainly translated from the SRMRpy package for batched processing with pytorch

from functools import lru_cache
from math import ceil
from typing import *

import torch
from torch import Tensor
from torch.nn.functional import pad

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _GAMMATONE_AVAILABEL, _TORCHAUDIO_AVAILABEL

if _TORCHAUDIO_AVAILABEL:
    from torchaudio.functional.filtering import lfilter
else:
    lfilter = None
    __doctest_skip__ = ["speech_reverberation_modulation_energy_ratio"]

if _GAMMATONE_AVAILABEL:
    from gammatone.fftweight import fft_gtgram
    from gammatone.filters import centre_freqs, make_erb_filters
else:
    fft_gtgram, centre_freqs, make_erb_filters = None, None, None
    __doctest_skip__ = ["speech_reverberation_modulation_energy_ratio"]


@lru_cache(maxsize=100)
def _calc_erbs(low_freq: float, fs: int, n_filters: int, device: torch.device) -> Tensor:
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1
    erbs = ((centre_freqs(fs, n_filters, low_freq) / ear_q) ** order + min_bw**order) ** (1 / order)
    erbs = torch.tensor(erbs, device=device)
    return erbs


@lru_cache(maxsize=100)
def _make_erb_filters(fs: int, num_freqs: int, cutoff: float, device: torch.device) -> Tensor:
    cfs = centre_freqs(fs, num_freqs, cutoff)
    fcoefs = make_erb_filters(fs, cfs)
    fcoefs = torch.tensor(fcoefs, device=device)
    return fcoefs


@lru_cache(maxsize=100)
def _compute_modulation_filterbank_and_cutoffs(
    min_cf: float, max_cf: float, n: int, Q: int, fs: float, q: int, device: torch.device
) -> Union[Tensor, Tensor, Tensor, Tensor]:
    # this function is translated from the SRMRpy packaged
    spacing_factor = (max_cf / min_cf) ** (1.0 / (n - 1))
    cfs = torch.zeros(n, dtype=torch.float64)
    cfs[0] = min_cf
    for k in range(1, n):
        cfs[k] = cfs[k - 1] * spacing_factor

    def _make_modulation_filter(w0, Q):
        W0 = torch.tan(w0 / 2)
        B0 = W0 / Q
        b = torch.tensor([B0, 0, -B0], dtype=torch.float64)
        a = torch.tensor([(1 + B0 + W0**2), (2 * W0**2 - 2), (1 - B0 + W0**2)], dtype=torch.float64)
        return torch.stack([b, a], dim=0)

    mfb = torch.stack([_make_modulation_filter(w0, Q) for w0 in 2 * torch.pi * cfs / fs], dim=0)

    def _calc_cutoffs(cfs, fs, q):
        # Calculates cutoff frequencies (3 dB) for 2nd order bandpass
        w0 = 2 * torch.pi * cfs / fs
        B0 = torch.tan(w0 / 2) / q
        L = cfs - (B0 * fs / (2 * torch.pi))
        R = cfs + (B0 * fs / (2 * torch.pi))
        return L, R

    cfs = cfs.to(device=device)
    mfb = mfb.to(device=device)
    L, R = _calc_cutoffs(cfs, fs, q)
    return cfs, mfb, L, R


def _hilbert(x: Tensor, N: int = None):
    if x.is_complex():
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[-1]
        # Make N multiple of 16 to make sure the transform will be fast
        if N % 16:
            N = ceil(N / 16) * 16
    if N <= 0:
        raise ValueError("N must be positive.")

    x_fft = torch.fft.fft(x, n=N, dim=-1)
    h = torch.zeros(N, dtype=x.dtype, device=x.device, requires_grad=False)
    assert N % 2 == 0, N
    h[0] = h[N // 2] = 1
    h[1 : N // 2] = 2

    y = torch.fft.ifft(x_fft * h, dim=-1)
    y = y[..., : x.shape[-1]]
    return y


def _erb_filterbank(wave: Tensor, coefs: Tensor) -> Tensor:
    """Translated from gammatone package.

    Args:
        wave: shape [B, time]
        coefs: shape [N, 10]

    Returns:
        Tensor: shape [B, N, time]
    """
    n_batch, time = wave.shape
    wave = wave.to(dtype=coefs.dtype).reshape(n_batch, 1, time)  # [B, time]
    wave = wave.expand(-1, coefs.shape[0], -1)  # [B, N, time]

    gain = coefs[:, 9]
    As1 = coefs[:, (0, 1, 5)]  # A0, A11, A2
    As2 = coefs[:, (0, 2, 5)]  # A0, A12, A2
    As3 = coefs[:, (0, 3, 5)]  # A0, A13, A2
    As4 = coefs[:, (0, 4, 5)]  # A0, A14, A2
    Bs = coefs[:, 6:9]  # B0, B1, B2

    y1 = lfilter(wave, Bs, As1, batching=True)
    y2 = lfilter(y1, Bs, As2, batching=True)
    y3 = lfilter(y2, Bs, As3, batching=True)
    y4 = lfilter(y3, Bs, As4, batching=True)
    output = y4 / gain.reshape(1, -1, 1)

    return output


def _normalize_energy(energy: Tensor, drange: float = 30.0) -> Tensor:
    peak_energy = torch.mean(energy, dim=1, keepdim=True).max(dim=2, keepdim=True).values
    peak_energy = peak_energy.max(dim=3, keepdim=True).values
    min_energy = peak_energy * 10.0 ** (-drange / 10.0)
    energy = torch.where(energy < min_energy, min_energy, energy)
    energy = torch.where(energy > peak_energy, peak_energy, energy)
    return energy


def _cal_srmr_score(BW: Tensor, avg_energy: Tensor, cutoffs: Tensor) -> Tensor:
    if (cutoffs[4] <= BW) and (cutoffs[5] > BW):
        Kstar = 5
    elif (cutoffs[5] <= BW) and (cutoffs[6] > BW):
        Kstar = 6
    elif (cutoffs[6] <= BW) and (cutoffs[7] > BW):
        Kstar = 7
    elif cutoffs[7] <= BW:
        Kstar = 8
    return torch.sum(avg_energy[:, :4]) / torch.sum(avg_energy[:, 4:Kstar])


def speech_reverberation_modulation_energy_ratio(
    preds: Tensor,
    fs: int,
    n_cochlear_filters: int = 23,
    low_freq: float = 125,
    min_cf: float = 4,
    max_cf: Optional[float] = 128,
    norm: bool = False,
    fast: bool = False,
) -> Tensor:
    """Calculate `Speech-to-Reverberation Modulation Energy Ratio`_ (SRMR).

    SRMR is a non-intrusive metric for speech quality and intelligibility based on
    a modulation spectral representation of the speech signal.
    This code is translated from `SRMRToolbox`_ and `SRMRpy`_.

    Args:
        preds: shape ``(..., time)``
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
            then 30 Hz will be used for `norm==False`, otherwise 128 Hz will be used.
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.
            Note: this argument is inherited from `SRMRpy`_. As the translated code is based to pytorch,
            setting `fast=True` may slow down the speed for calculating this metric on GPU.

    .. note:: using this metrics requires you to have ``gammatone`` and ``torchaudio`` installed.
        Either install as ``pip install torchmetrics[audio]`` or ``pip install torchaudio``
        and ``pip install git+https://github.com/detly/gammatone``.

    Returns:
        Tensor: srmr value, shape ``(...)``

    Raises:
        ModuleNotFoundError:
            If ``gammatone`` or ``torchaudio`` package is not installed

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import speech_reverberation_modulation_energy_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> speech_reverberation_modulation_energy_ratio(preds, 8000)
        tensor([0.3354], dtype=torch.float64)
    """
    if not _TORCHAUDIO_AVAILABEL or not _GAMMATONE_AVAILABEL:
        raise ModuleNotFoundError(
            "speech_reverberation_modulation_energy_ratio requires you to have `gammatone` and"
            " `torchaudio` installed. Either install as ``pip install torchmetrics[audio]`` or "
            "``pip install torchaudio`` and ``pip install git+https://github.com/detly/gammatone``"
        )

    shape = preds.shape
    if len(shape) == 1:
        preds = preds.reshape(1, -1)  # [B, time]
    else:
        preds = preds.reshape(-1, shape[-1])  # [B, time]
    n_batch, time = preds.shape
    # convert int type to float
    if not torch.is_floating_point(preds):
        preds = preds.to(torch.float64) / torch.finfo(preds.dtype).max

    # norm values in preds to [-1, 1], as lfilter requires an input in this range
    max_vals = preds.abs().max(dim=-1, keepdim=True).values
    val_norm = torch.where(max_vals > 1, max_vals, 1)
    preds = preds / val_norm

    wLengthS = 0.256
    wIncS = 0.064
    # Computing gammatone envelopes
    if fast:
        rank_zero_warn("`fast=True` may slow down the speed of SRMR metric on GPU.")
        mfs = 400.0
        temp = []
        preds_np = preds.detach().cpu().numpy()
        for b in range(n_batch):
            gt_env_b = fft_gtgram(preds_np[b], fs, 0.010, 0.0025, n_cochlear_filters, low_freq)
            temp.append(torch.tensor(gt_env_b))
        gt_env = torch.stack(temp, dim=0).to(device=preds.device)
    else:
        fcoefs = _make_erb_filters(fs, n_cochlear_filters, low_freq, device=preds.device)  # [N_filters, 10]
        gt_env = torch.abs(_hilbert(_erb_filterbank(preds, fcoefs)))  # [B, N_filters, time]
        mfs = fs

    wLength = ceil(wLengthS * mfs)
    wInc = ceil(wIncS * mfs)

    # Computing modulation filterbank with Q = 2 and 8 channels
    if max_cf is None:
        max_cf = 30 if norm else 128
    _, MF, cutoffs, _ = _compute_modulation_filterbank_and_cutoffs(
        min_cf, max_cf, n=8, Q=2, fs=mfs, q=2, device=preds.device
    )

    n_frames = int(1 + (time - wLength) // wInc)
    w = torch.hamming_window(wLength + 1, dtype=torch.float64, device=preds.device)[:-1]
    mod_out = lfilter(
        gt_env.unsqueeze(-2).expand(-1, -1, MF.shape[0], -1), MF[:, 1, :], MF[:, 0, :], clamp=False, batching=True
    )  # [B, N_filters, 8, time]
    # pad signal if it's shorter than window or it is not multiple of wInc
    padding = (0, max(ceil(time / wInc) * wInc - time, wLength - time))
    mod_out_pad = pad(mod_out, pad=padding, mode="constant", value=0)
    mod_out_frame = mod_out_pad.unfold(-1, wLength, wInc)
    energy = ((mod_out_frame[..., :n_frames, :] * w) ** 2).sum(dim=-1)  # [B, N_filters, 8, n_frames]

    if norm:
        energy = _normalize_energy(energy)

    erbs = torch.flipud(_calc_erbs(low_freq, fs, n_cochlear_filters, device=preds.device))

    avg_energy = torch.mean(energy, dim=-1)
    total_energy = torch.sum(avg_energy.reshape(n_batch, -1), dim=-1)
    AC_energy = torch.sum(avg_energy, axis=2)
    AC_perc = AC_energy * 100 / total_energy.reshape(-1, 1)
    AC_perc_cumsum = AC_perc.flip(-1).cumsum(-1)
    K90perc_idx = torch.nonzero((AC_perc_cumsum > 90).cumsum(-1) == 1)[:, 1]
    BW = erbs[K90perc_idx]

    temp = []
    for b in range(n_batch):
        score = _cal_srmr_score(BW[b], avg_energy[b], cutoffs=cutoffs)
        temp.append(score)
    score = torch.stack(temp)

    if len(shape) > 1:  # recover original shape
        score = score.reshape(*shape[:-1])

    return score
