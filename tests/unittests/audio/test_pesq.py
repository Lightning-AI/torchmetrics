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
from functools import partial

import pytest
import torch
from pesq import pesq as pesq_backend
from scipy.io import wavfile
from torch import Tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality

from unittests import _Input
from unittests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


# for 8k sample rate, need at least 8k/4=2000 samples
inputs_8k = _Input(
    preds=torch.rand(2, 3, 2100),
    target=torch.rand(2, 3, 2100),
)
# for 16k sample rate, need at least 16k/4=4000 samples
inputs_16k = _Input(
    preds=torch.rand(2, 3, 4100),
    target=torch.rand(2, 3, 4100),
)


def _pesq_original_batch(preds: Tensor, target: Tensor, fs: int, mode: str):
    """Comparison function."""
    # shape: preds [BATCH_SIZE, Time] , target [BATCH_SIZE, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, Time] , target [NUM_BATCHES*BATCH_SIZE, Time]
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        pesq_val = pesq_backend(fs, target[b, ...], preds[b, ...], mode)
        mss.append(pesq_val)
    return torch.tensor(mss)


def _average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


pesq_original_batch_8k_nb = partial(_pesq_original_batch, fs=8000, mode="nb")
pesq_original_batch_16k_nb = partial(_pesq_original_batch, fs=16000, mode="nb")
pesq_original_batch_16k_wb = partial(_pesq_original_batch, fs=16000, mode="wb")


@pytest.mark.parametrize(
    "preds, target, ref_metric, fs, mode",
    [
        (inputs_8k.preds, inputs_8k.target, pesq_original_batch_8k_nb, 8000, "nb"),
        (inputs_16k.preds, inputs_16k.target, pesq_original_batch_16k_nb, 16000, "nb"),
        (inputs_16k.preds, inputs_16k.target, pesq_original_batch_16k_wb, 16000, "wb"),
    ],
)
class TestPESQ(MetricTester):
    """Test class for `PerceptualEvaluationSpeechQuality` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("num_processes", [1, 2])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_pesq(self, preds, target, ref_metric, fs, mode, num_processes, ddp):
        """Test class implementation of metric."""
        if num_processes != 1 and ddp:
            pytest.skip("Multiprocessing and ddp does not work together")
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PerceptualEvaluationSpeechQuality,
            reference_metric=partial(_average_metric, metric_func=ref_metric),
            metric_args={"fs": fs, "mode": mode, "n_processes": num_processes},
        )

    @pytest.mark.parametrize("num_processes", [1, 2])
    def test_pesq_functional(self, preds, target, ref_metric, fs, mode, num_processes):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            perceptual_evaluation_speech_quality,
            ref_metric,
            metric_args={"fs": fs, "mode": mode, "n_processes": num_processes},
        )

    def test_pesq_differentiability(self, preds, target, ref_metric, fs, mode):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=perceptual_evaluation_speech_quality,
            metric_args={"fs": fs, "mode": mode},
        )

    def test_pesq_half_cpu(self, preds, target, ref_metric, fs, mode):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("PESQ metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pesq_half_gpu(self, preds, target, ref_metric, fs, mode):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=partial(perceptual_evaluation_speech_quality, fs=fs, mode=mode),
            metric_args={"fs": fs, "mode": mode},
        )


def test_error_on_different_shape(metric_class=PerceptualEvaluationSpeechQuality):
    """Test that an error is raised on different shapes of input."""
    metric = metric_class(16000, "nb")
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_on_real_audio():
    """Test that metric works as expected on real audio signals."""
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    pesq = perceptual_evaluation_speech_quality(torch.from_numpy(deg), torch.from_numpy(ref), rate, "wb")
    assert pesq == 1.0832337141036987
    pesq = perceptual_evaluation_speech_quality(torch.from_numpy(deg), torch.from_numpy(ref), rate, "nb")
    assert pesq == 1.6072081327438354
