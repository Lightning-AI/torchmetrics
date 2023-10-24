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
import speechmetrics
import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

NUM_SAMPLES = 100


inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
)

speechmetrics_sisdr = speechmetrics.load("sisdr")


def _speechmetrics_si_sdr(preds: Tensor, target: Tensor, zero_mean: bool):
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
            metric = speechmetrics_sisdr(preds[i, j], target[i, j], rate=16000)
            ms.append(metric["sisdr"][0])
        mss.append(ms)
    return torch.tensor(mss)


def _average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


speechmetrics_si_sdr_zero_mean = partial(_speechmetrics_si_sdr, zero_mean=True)
speechmetrics_si_sdr_no_zero_mean = partial(_speechmetrics_si_sdr, zero_mean=False)


@pytest.mark.parametrize(
    "preds, target, ref_metric, zero_mean",
    [
        (inputs.preds, inputs.target, speechmetrics_si_sdr_zero_mean, True),
        (inputs.preds, inputs.target, speechmetrics_si_sdr_no_zero_mean, False),
    ],
)
class TestSISDR(MetricTester):
    """Test class for `ScaleInvariantSignalDistortionRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_si_sdr(self, preds, target, ref_metric, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ScaleInvariantSignalDistortionRatio,
            reference_metric=partial(_average_metric, metric_func=ref_metric),
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_functional(self, preds, target, ref_metric, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            scale_invariant_signal_distortion_ratio,
            ref_metric,
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SI-SDR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=ScaleInvariantSignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
