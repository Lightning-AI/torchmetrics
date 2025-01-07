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

import numpy as np
import pytest
import torch
from torch import Tensor

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_SAMPLES = 100


inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
)


class _SpeechMetricsSISDR:
    """The code from speechmetrics."""

    def __init__(
        self,
    ) -> None: ...

    def _test_window(self, audios, rate):
        # as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
        # and one estimate.
        # see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
        eps = np.finfo(audios[0].dtype).eps
        reference = audios[1].reshape(audios[1].size, 1)
        estimate = audios[0].reshape(audios[0].size, 1)
        rss = np.dot(reference.T, reference)

        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        sss = (e_true**2).sum()
        snn = (e_res**2).sum()

        return {"sisdr": 10 * np.log10((eps + sss) / (eps + snn))}


speechmetrics_sisdr = _SpeechMetricsSISDR()


def _reference_speechmetrics_si_sdr(preds: Tensor, target: Tensor, zero_mean: bool = True, reduce_mean: bool = False):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    if zero_mean:
        preds = preds - preds.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            metric = speechmetrics_sisdr._test_window([preds[i, j], target[i, j]], rate=16000)
            ms.append(metric["sisdr"])
        mss.append(ms)
    si_sdr = torch.tensor(mss)
    if reduce_mean:
        # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
        # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
        return si_sdr.mean()
    return si_sdr


@pytest.mark.parametrize(
    "preds, target, ref_metric",
    [(inputs.preds, inputs.target, _reference_speechmetrics_si_sdr)],
)
class TestSISNR(MetricTester):
    """Test class for `ScaleInvariantSignalNoiseRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_si_snr(self, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ScaleInvariantSignalNoiseRatio,
            reference_metric=partial(_reference_speechmetrics_si_sdr, reduce_mean=True),
        )

    def test_si_snr_functional(self, preds, target, ref_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            scale_invariant_signal_noise_ratio,
            ref_metric,
        )

    def test_si_snr_differentiability(self, preds, target, ref_metric):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalNoiseRatio,
            metric_functional=scale_invariant_signal_noise_ratio,
        )

    def test_si_snr_half_cpu(self, preds, target, ref_metric):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SI-SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_si_snr_half_gpu(self, preds, target, ref_metric):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalNoiseRatio,
            metric_functional=scale_invariant_signal_noise_ratio,
        )


def test_error_on_different_shape(metric_class=ScaleInvariantSignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
