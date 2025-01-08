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
import os
from functools import partial
from typing import Any, Optional

import numpy as np
import pytest
import torch
from torch import Tensor

from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.functional.audio.dnsmos import (
    DNSMOS_DIR,
    _load_session,
    deep_noise_suppression_mean_opinion_score,
)
from torchmetrics.utilities.imports import (
    _LIBROSA_AVAILABLE,
    _ONNXRUNTIME_AVAILABLE,
    _REQUESTS_AVAILABLE,
)
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

if _LIBROSA_AVAILABLE and _ONNXRUNTIME_AVAILABLE and _REQUESTS_AVAILABLE:
    import librosa
    import onnxruntime as ort
else:
    librosa, ort = None, None  # type:ignore

    class InferenceSession:  # type:ignore
        """Dummy InferenceSession."""

        def __init__(self, **kwargs: dict[str, Any]) -> None: ...


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
seed_all(42)


class _ComputeScore:
    """The implementation from DNS-Challenge."""

    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(os.path.expanduser(primary_model_path))
        self.p808_onnx_sess = ort.InferenceSession(os.path.expanduser(p808_model_path))

    def _audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def _get_polyfit_val(self, sig, bak, ovr, is_personalized):
        if is_personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized) -> dict[str, Any]:
        fs = SAMPLING_RATE
        audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs) if input_fs != fs else aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(self._audio_melspec(audio=audio_seg[:-160])).astype("float32")[
                np.newaxis, :, :
            ]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self._get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        return {
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
            "P808_MOS": np.mean(predicted_p808_mos),
        }


def _reference_metric_batch(
    preds: Tensor,  # shape:[BATCH_SIZE, Time]
    target: Tensor,  # for tester
    fs: int,
    personalized: bool,
    device: Optional[str] = None,  # for tester
    reduce_mean: bool = False,
    **kwargs: dict[str, Any],  # for tester
):
    # download onnx first
    _load_session(f"{DNSMOS_DIR}/{'p' if personalized else ''}DNSMOS/sig_bak_ovr.onnx", torch.device("cpu"))
    _load_session(f"{DNSMOS_DIR}/DNSMOS/model_v8.onnx", torch.device("cpu"))
    # construct ComputeScore
    cs = _ComputeScore(
        f"{DNSMOS_DIR}/{'p' if personalized else ''}DNSMOS/sig_bak_ovr.onnx",
        f"{DNSMOS_DIR}/DNSMOS/model_v8.onnx",
    )

    shape = preds.shape
    preds = preds.reshape(1, -1) if len(shape) == 1 else preds.reshape(-1, shape[-1])

    preds = preds.detach().cpu().numpy()
    score = []
    for b in range(preds.shape[0]):
        val = cs.__call__(preds[b, ...], fs, personalized)
        score.append([val["P808_MOS"], val["SIG"], val["BAK"], val["OVRL"]])
    score = torch.tensor(score)
    if reduce_mean:
        # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
        # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
        return score.mean(dim=0)
    return score.reshape(*shape[:-1], 4).reshape(shape[:-1] + (4,)).numpy()


def _dnsmos_cheat(preds, target, **kwargs: dict[str, Any]):
    # cheat the MetricTester as the deep_noise_suppression_mean_opinion_score doesn't need target
    return deep_noise_suppression_mean_opinion_score(preds, **kwargs)


class _DNSMOSCheat(DeepNoiseSuppressionMeanOpinionScore):
    # cheat the MetricTester as DeepNoiseSuppressionMeanOpinionScore doesn't need target
    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds=preds)


preds = torch.rand(2, 2, 8000)


@pytest.mark.parametrize(
    "preds, fs, personalized",
    [
        (preds, 8000, False),
        (preds, 8000, True),
        (preds, 16000, False),
        (preds, 16000, True),
    ],
)
class TestDNSMOS(MetricTester):
    """Test class for `DeepNoiseSuppressionMeanOpinionScore` metric."""

    atol = 5e-3

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_dnsmos(self, preds: Tensor, fs: int, personalized: bool, ddp: bool, device=None):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_DNSMOSCheat,
            reference_metric=partial(
                _reference_metric_batch,
                fs=fs,
                personalized=personalized,
                device=device,
                reduce_mean=True,
            ),
            metric_args={"fs": fs, "personalized": personalized, "device": device},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_dnsmos_cuda(self, preds: Tensor, fs: int, personalized: bool, ddp: bool, device="cuda:0"):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_DNSMOSCheat,
            reference_metric=partial(
                _reference_metric_batch,
                fs=fs,
                personalized=personalized,
                device=device,
                reduce_mean=True,
            ),
            metric_args={"fs": fs, "personalized": personalized, "device": device},
        )

    def test_dnsmos_functional(self, preds: Tensor, fs: int, personalized: bool, device="cpu"):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_dnsmos_cheat,
            reference_metric=partial(
                _reference_metric_batch,
                fs=fs,
                personalized=personalized,
                device=device,
                reduce_mean=False,
            ),
            metric_args={"fs": fs, "personalized": personalized, "device": device},
        )
