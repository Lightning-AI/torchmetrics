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

# Note: without special mention, the functions in this file are mainly translated from
# the SRMRpy package for batched processing with pytorch

from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import pad

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
    _GAMMATONE_AVAILABEL,
    _TORCHAUDIO_AVAILABEL,
    _TORCHAUDIO_GREATER_EQUAL_0_10,
)

if _TORCHAUDIO_AVAILABEL and _TORCHAUDIO_GREATER_EQUAL_0_10:
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
    return torch.tensor(erbs, device=device)


@lru_cache(maxsize=100)
def _make_erb_filters(fs: int, num_freqs: int, cutoff: float, device: torch.device) -> Tensor:
    cfs = centre_freqs(fs, num_freqs, cutoff)
    fcoefs = make_erb_filters(fs, cfs)
    return torch.tensor(fcoefs, device=device)


@lru_cache(maxsize=100)
def _compute_modulation_filterbank_and_cutoffs(
    min_cf: float, max_cf: float, n: int, fs: float, q: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # this function is translated from the SRMRpy packaged
    spacing_factor = (max_cf / min_cf) ** (1.0 / (n - 1))
    cfs = torch.zeros(n, dtype=torch.float64)
    cfs[0] = min_cf
    for k in range(1, n):
        cfs[k] = cfs[k - 1] * spacing_factor

    def _make_modulation_filter(w0: Tensor, q: int) -> Tensor:
        w0 = torch.tan(w0 / 2)
        b0 = w0 / q
        b = torch.tensor([b0, 0, -b0], dtype=torch.float64)
        a = torch.tensor([(1 + b0 + w0**2), (2 * w0**2 - 2), (1 - b0 + w0**2)], dtype=torch.float64)
        return torch.stack([b, a], dim=0)

    mfb = torch.stack([_make_modulation_filter(w0, q) for w0 in 2 * pi * cfs / fs], dim=0)

    def _calc_cutoffs(cfs: Tensor, fs: float, q: int) -> Tuple[Tensor, Tensor]:
        # Calculates cutoff frequencies (3 dB) for 2nd order bandpass
        w0 = 2 * pi * cfs / fs
        b0 = torch.tan(w0 / 2) / q
        ll = cfs - (b0 * fs / (2 * pi))
        rr = cfs + (b0 * fs / (2 * pi))
        return ll, rr

    cfs = cfs.to(device=device)
    mfb = mfb.to(device=device)
    ll, rr = _calc_cutoffs(cfs, fs, q)
    return cfs, mfb, ll, rr


def _hilbert(x: Tensor, n: Optional[int] = None) -> Tensor:
    if x.is_complex():
        raise ValueError("x must be real.")
    if n is None:
        n = x.shape[-1]
        # Make N multiple of 16 to make sure the transform will be fast
        if n % 16:
            n = ceil(n / 16) * 16
    if n <= 0:
        raise ValueError("N must be positive.")

    x_fft = torch.fft.fft(x, n=n, dim=-1)
    h = torch.zeros(n, dtype=x.dtype, device=x.device, requires_grad=False)

    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    y = torch.fft.ifft(x_fft * h, dim=-1)
    return y[..., : x.shape[-1]]


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
    as1 = coefs[:, (0, 1, 5)]  # A0, A11, A2
    as2 = coefs[:, (0, 2, 5)]  # A0, A12, A2
    as3 = coefs[:, (0, 3, 5)]  # A0, A13, A2
    as4 = coefs[:, (0, 4, 5)]  # A0, A14, A2
    bs = coefs[:, 6:9]  # B0, B1, B2

    y1 = lfilter(wave, bs, as1, batching=True)
    y2 = lfilter(y1, bs, as2, batching=True)
    y3 = lfilter(y2, bs, as3, batching=True)
    y4 = lfilter(y3, bs, as4, batching=True)
    return y4 / gain.reshape(1, -1, 1)


def _normalize_energy(energy: Tensor, drange: float = 30.0) -> Tensor:
    """Normalize energy to a dynamic range of 30 dB.

    Args:
        energy: shape [B, N_filters, 8, n_frames]
        drange: dynamic range in dB

    """
    peak_energy = torch.mean(energy, dim=1, keepdim=True).max(dim=2, keepdim=True).values
    peak_energy = peak_energy.max(dim=3, keepdim=True).values
    min_energy = peak_energy * 10.0 ** (-drange / 10.0)
    energy = torch.where(energy < min_energy, min_energy, energy)
    return torch.where(energy > peak_energy, peak_energy, energy)


def _cal_srmr_score(bw: Tensor, avg_energy: Tensor, cutoffs: Tensor) -> Tensor:
    """Calculate srmr score."""
    if (cutoffs[4] <= bw) and (cutoffs[5] > bw):
        kstar = 5
    elif (cutoffs[5] <= bw) and (cutoffs[6] > bw):
        kstar = 6
    elif (cutoffs[6] <= bw) and (cutoffs[7] > bw):
        kstar = 7
    elif cutoffs[7] <= bw:
        kstar = 8
    else:
        raise ValueError("Something wrong with the cutoffs compared to bw values.")
    return torch.sum(avg_energy[:, :4]) / torch.sum(avg_energy[:, 4:kstar])


def speech_reverberation_modulation_energy_ratio(
    preds: Tensor,
    fs: int,
    n_cochlear_filters: int = 23,
    low_freq: float = 125,
    min_cf: float = 4,
    max_cf: Optional[float] = None,
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

    .. note::
        This implementation is experimental, and might not be consistent with the matlab
        implementation `SRMRToolbox`_, especially the fast implementation.
        The slow versions, a) fast=False, norm=False, max_cf=128, b) fast=False, norm=True, max_cf=30, have
        a relatively small inconsistence.

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
    if not _TORCHAUDIO_AVAILABEL or not _TORCHAUDIO_GREATER_EQUAL_0_10 or not _GAMMATONE_AVAILABEL:
        raise ModuleNotFoundError(
            "speech_reverberation_modulation_energy_ratio requires you to have `gammatone` and"
            " `torchaudio>=0.10` installed. Either install as ``pip install torchmetrics[audio]`` or "
            "``pip install torchaudio>=0.10`` and ``pip install git+https://github.com/detly/gammatone``"
        )
    _srmr_arg_validate(
        fs=fs,
        n_cochlear_filters=n_cochlear_filters,
        low_freq=low_freq,
        min_cf=min_cf,
        max_cf=max_cf,
        norm=norm,
        fast=fast,
    )
    shape = preds.shape
    preds = preds.reshape(1, -1) if len(shape) == 1 else preds.reshape(-1, shape[-1])
    n_batch, time = preds.shape
    # convert int type to float
    if not torch.is_floating_point(preds):
        preds = preds.to(torch.float64) / torch.finfo(preds.dtype).max

    # norm values in preds to [-1, 1], as lfilter requires an input in this range
    max_vals = preds.abs().max(dim=-1, keepdim=True).values
    val_norm = torch.where(
        max_vals > 1,
        max_vals,
        torch.tensor(1.0, dtype=max_vals.dtype, device=max_vals.device),
    )
    preds = preds / val_norm

    w_length_s = 0.256
    w_inc_s = 0.064
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

    w_length = ceil(w_length_s * mfs)
    w_inc = ceil(w_inc_s * mfs)

    # Computing modulation filterbank with Q = 2 and 8 channels
    if max_cf is None:
        max_cf = 30 if norm else 128
    _, mf, cutoffs, _ = _compute_modulation_filterbank_and_cutoffs(
        min_cf, max_cf, n=8, fs=mfs, q=2, device=preds.device
    )

    n_frames = int(1 + (time - w_length) // w_inc)
    w = torch.hamming_window(w_length + 1, dtype=torch.float64, device=preds.device)[:-1]
    mod_out = lfilter(
        gt_env.unsqueeze(-2).expand(-1, -1, mf.shape[0], -1), mf[:, 1, :], mf[:, 0, :], clamp=False, batching=True
    )  # [B, N_filters, 8, time]
    # pad signal if it's shorter than window or it is not multiple of wInc
    padding = (0, max(ceil(time / w_inc) * w_inc - time, w_length - time))
    mod_out_pad = pad(mod_out, pad=padding, mode="constant", value=0)
    mod_out_frame = mod_out_pad.unfold(-1, w_length, w_inc)
    energy = ((mod_out_frame[..., :n_frames, :] * w) ** 2).sum(dim=-1)  # [B, N_filters, 8, n_frames]

    if norm:
        energy = _normalize_energy(energy)

    erbs = torch.flipud(_calc_erbs(low_freq, fs, n_cochlear_filters, device=preds.device))

    avg_energy = torch.mean(energy, dim=-1)
    total_energy = torch.sum(avg_energy.reshape(n_batch, -1), dim=-1)
    ac_energy = torch.sum(avg_energy, dim=2)
    ac_perc = ac_energy * 100 / total_energy.reshape(-1, 1)
    ac_perc_cumsum = ac_perc.flip(-1).cumsum(-1)
    k90perc_idx = torch.nonzero((ac_perc_cumsum > 90).cumsum(-1) == 1)[:, 1]
    bw = erbs[k90perc_idx]

    temp = []
    for b in range(n_batch):
        score = _cal_srmr_score(bw[b], avg_energy[b], cutoffs=cutoffs)
        temp.append(score)
    score = torch.stack(temp)

    return score.reshape(*shape[:-1]) if len(shape) > 1 else score  # recover original shape


def _srmr_arg_validate(
    fs: int,
    n_cochlear_filters: int = 23,
    low_freq: float = 125,
    min_cf: float = 4,
    max_cf: Optional[float] = 128,
    norm: bool = False,
    fast: bool = False,
) -> None:
    """Validate the arguments for speech_reverberation_modulation_energy_ratio.

    Args:
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.

    """
    if not (isinstance(fs, int) and fs > 0):
        raise ValueError(f"Expected argument `fs` to be an int larger than 0, but got {fs}")
    if not (isinstance(n_cochlear_filters, int) and n_cochlear_filters > 0):
        raise ValueError(
            f"Expected argument `n_cochlear_filters` to be an int larger than 0, but got {n_cochlear_filters}"
        )
    if not ((isinstance(low_freq, (float, int))) and low_freq > 0):
        raise ValueError(f"Expected argument `low_freq` to be a float larger than 0, but got {low_freq}")
    if not ((isinstance(min_cf, (float, int))) and min_cf > 0):
        raise ValueError(f"Expected argument `min_cf` to be a float larger than 0, but got {min_cf}")
    if max_cf is not None and not ((isinstance(max_cf, (float, int))) and max_cf > 0):
        raise ValueError(f"Expected argument `max_cf` to be a float larger than 0, but got {max_cf}")
    if not isinstance(norm, bool):
        raise ValueError("Expected argument `norm` to be a bool value")
    if not isinstance(fast, bool):
        raise ValueError("Expected argument `fast` to be a bool value")
