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
from collections import namedtuple
from functools import partial

import pytest
import torch
from torch import Tensor
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalNoiseRatio,
    SourceAggregatedSignalDistortionRatio,
)
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    signal_noise_ratio,
    source_aggregated_signal_distortion_ratio,
)

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Time = 100

Input = namedtuple("Input", ["preds", "target"])

inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, Time),
)


def _ref_metric(preds: Tensor, target: Tensor, scale_invariant: bool, zero_mean: bool):
    # According to the original paper, the sa-sdr equals to si-sdr with inputs concatenated over the speaker
    # dimension if scale_invariant==True. Accordingly, for scale_invariant==False, the sa-sdr equals to snr.
    # shape: preds [BATCH_SIZE, Spk, Time] , target [BATCH_SIZE, Spk, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, Spk, Time], target [NUM_BATCHES*BATCH_SIZE, Spk, Time]

    preds = preds.reshape(preds.shape[0], preds[1] * preds[2])
    target = target.reshape(preds.shape[0], preds[1] * preds[2])
    if scale_invariant:
        return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=zero_mean)
    else:
        return signal_noise_ratio(preds=preds, target=target, zero_mean=zero_mean)


def _average_metric(**kwargs):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return _ref_metric(**kwargs).mean()


@pytest.mark.parametrize(
    "preds, target, scale_invariant, zero_mean",
    [
        (inputs.preds, inputs.target, True, False),
        (inputs.preds, inputs.target, True, True),
        (inputs.preds, inputs.target, False, False),
        (inputs.preds, inputs.target, False, True),
    ],
)
class TestSASDR(MetricTester):
    """Test class for `SourceAggregatedSignalDistortionRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_si_sdr(self, preds, target, scale_invariant, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SourceAggregatedSignalDistortionRatio,
            reference_metric=_average_metric,
            metric_args={
                "scale_invariant": scale_invariant,
                "zero_mean": zero_mean,
            },
        )

    def test_sa_sdr_functional(self, preds, target, scale_invariant, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            source_aggregated_signal_distortion_ratio,
            reference_metric=_ref_metric,
            metric_args={
                "scale_invariant": scale_invariant,
                "zero_mean": zero_mean,
            },
        )

    def test_sa_sdr_differentiability(self, preds, target, scale_invariant, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SourceAggregatedSignalDistortionRatio,
            metric_functional=source_aggregated_signal_distortion_ratio,
            metric_args={
                "scale_invariant": scale_invariant,
                "zero_mean": zero_mean,
            },
        )

    def test_sa_sdr_half_cpu(self, preds, target, scale_invariant, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SA-SDR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sa_sdr_half_gpu(self, preds, target, scale_invariant, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SourceAggregatedSignalDistortionRatio,
            metric_functional=source_aggregated_signal_distortion_ratio,
            metric_args={
                "scale_invariant": scale_invariant,
                "zero_mean": zero_mean,
            },
        )


def test_error_on_shape(metric_class=SourceAggregatedSignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))

    with pytest.raises(RuntimeError, match="The preds and target should have the shape (..., spk, time)*"):
        metric(torch.randn(100), torch.randn(100))
