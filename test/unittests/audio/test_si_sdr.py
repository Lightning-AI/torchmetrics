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
import speechmetrics
import torch
from torch import Tensor

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.helpers import seed_all
from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester

seed_all(42)

Time = 100

Input = namedtuple("Input", ["preds", "target"])

inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
)

speechmetrics_sisdr = speechmetrics.load("sisdr")


def speechmetrics_si_sdr(preds: Tensor, target: Tensor, zero_mean: bool):
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


def average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


speechmetrics_si_sdr_zero_mean = partial(speechmetrics_si_sdr, zero_mean=True)
speechmetrics_si_sdr_no_zero_mean = partial(speechmetrics_si_sdr, zero_mean=False)


@pytest.mark.parametrize(
    "preds, target, sk_metric, zero_mean",
    [
        (inputs.preds, inputs.target, speechmetrics_si_sdr_zero_mean, True),
        (inputs.preds, inputs.target, speechmetrics_si_sdr_no_zero_mean, False),
    ],
)
class TestSISDR(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_si_sdr(self, preds, target, sk_metric, zero_mean, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ScaleInvariantSignalDistortionRatio,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_si_sdr_functional(self, preds, target, sk_metric, zero_mean):
        self.run_functional_metric_test(
            preds,
            target,
            scale_invariant_signal_distortion_ratio,
            sk_metric,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_si_sdr_differentiability(self, preds, target, sk_metric, zero_mean):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_si_sdr_half_cpu(self, preds, target, sk_metric, zero_mean):
        pytest.xfail("SI-SDR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_si_sdr_half_gpu(self, preds, target, sk_metric, zero_mean):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=ScaleInvariantSignalDistortionRatio):
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
