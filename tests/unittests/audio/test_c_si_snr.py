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

import pytest
import torch
from scipy.io import wavfile
from torchmetrics.audio import ComplexScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import complex_scale_invariant_signal_noise_ratio

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 129, 20, 2),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 129, 20, 2),
)


@pytest.mark.parametrize(
    "preds, target, ref_metric, zero_mean",
    [
        (inputs.preds, inputs.target, None, True),
        (inputs.preds, inputs.target, None, False),
    ],
)
class TestComplexSISNR(MetricTester):
    """Test class for `ComplexScaleInvariantSignalNoiseRatio` metric."""

    atol = 1e-2

    def test_c_si_snr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ComplexScaleInvariantSignalNoiseRatio,
            metric_functional=complex_scale_invariant_signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_c_si_sdr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("C-SI-SDR metric does not support cpu + half precision")

    def test_c_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        pytest.xfail("C-SI-SDR metric does not support gpu + half precision")


def test_on_real_audio():
    """Test that metric works as expected on real audio signals."""
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    ref = torch.tensor(ref, dtype=torch.float32)
    deg = torch.tensor(deg, dtype=torch.float32)
    ref_stft = torch.stft(ref, n_fft=256, hop_length=128, return_complex=True)
    deg_stft = torch.stft(deg, n_fft=256, hop_length=128, return_complex=True)

    v = complex_scale_invariant_signal_noise_ratio(deg_stft, ref_stft, zero_mean=False)
    assert torch.allclose(v, torch.tensor(0.03019072115421295, dtype=v.dtype), atol=1e-4), v
    v = complex_scale_invariant_signal_noise_ratio(deg_stft, ref_stft, zero_mean=True)
    assert torch.allclose(v, torch.tensor(0.030391741544008255, dtype=v.dtype), atol=1e-4), v


def test_error_on_incorrect_shape(metric_class=ComplexScaleInvariantSignalNoiseRatio):
    """Test that error is raised on incorrect shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the shape (..., frequency, time, 2)*",
    ):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_different_shape(metric_class=ComplexScaleInvariantSignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape*"):
        metric(torch.randn(129, 100, 2), torch.randn(129, 101, 2))
