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
from collections import namedtuple
from functools import partial

import pytest
import torch
from pesq import pesq as pesq_backend
from scipy.io import wavfile
from torch import Tensor

from tests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB
from tests.helpers import seed_all
from tests.helpers.testers import MetricTester
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

# for 8k sample rate, need at least 8k/4=2000 samples
inputs_8k = Input(
    preds=torch.rand(2, 3, 2100),
    target=torch.rand(2, 3, 2100),
)
# for 16k sample rate, need at least 16k/4=4000 samples
inputs_16k = Input(
    preds=torch.rand(2, 3, 4100),
    target=torch.rand(2, 3, 4100),
)


def pesq_original_batch(preds: Tensor, target: Tensor, fs: int, mode: str):
    # shape: preds [BATCH_SIZE, Time] , target [BATCH_SIZE, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, Time] , target [NUM_BATCHES*BATCH_SIZE, Time]
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        pesq_val = pesq_backend(fs, target[b, ...], preds[b, ...], mode)
        mss.append(pesq_val)
    return torch.tensor(mss)


def average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


pesq_original_batch_8k_nb = partial(pesq_original_batch, fs=8000, mode="nb")
pesq_original_batch_16k_nb = partial(pesq_original_batch, fs=16000, mode="nb")
pesq_original_batch_16k_wb = partial(pesq_original_batch, fs=16000, mode="wb")


@pytest.mark.parametrize(
    "preds, target, sk_metric, fs, mode",
    [
        (inputs_8k.preds, inputs_8k.target, pesq_original_batch_8k_nb, 8000, "nb"),
        (inputs_16k.preds, inputs_16k.target, pesq_original_batch_16k_nb, 16000, "nb"),
        (inputs_16k.preds, inputs_16k.target, pesq_original_batch_16k_wb, 16000, "wb"),
    ],
)
class TestPESQ(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_pesq(self, preds, target, sk_metric, fs, mode, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PerceptualEvaluationSpeechQuality,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(fs=fs, mode=mode),
        )

    def test_pesq_functional(self, preds, target, sk_metric, fs, mode):
        self.run_functional_metric_test(
            preds,
            target,
            perceptual_evaluation_speech_quality,
            sk_metric,
            metric_args=dict(fs=fs, mode=mode),
        )

    def test_pesq_differentiability(self, preds, target, sk_metric, fs, mode):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=perceptual_evaluation_speech_quality,
            metric_args=dict(fs=fs, mode=mode),
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_pesq_half_cpu(self, preds, target, sk_metric, fs, mode):
        pytest.xfail("PESQ metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pesq_half_gpu(self, preds, target, sk_metric, fs, mode):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=partial(perceptual_evaluation_speech_quality, fs=fs, mode=mode),
            metric_args=dict(fs=fs, mode=mode),
        )


def test_error_on_different_shape(metric_class=PerceptualEvaluationSpeechQuality):
    metric = metric_class(16000, "nb")
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_on_real_audio():
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    pesq = perceptual_evaluation_speech_quality(torch.from_numpy(deg), torch.from_numpy(ref), rate, "wb")
    assert pesq == 1.0832337141036987
    pesq = perceptual_evaluation_speech_quality(torch.from_numpy(deg), torch.from_numpy(ref), rate, "nb")
    assert pesq == 1.6072081327438354
