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
from typing import Callable

import pytest
import torch
from mir_eval.separation import bss_eval_images as mir_eval_bss_eval_images
from torch import Tensor
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.functional.audio import signal_noise_ratio

from unittests import _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


inputs = _Input(
    preds=torch.rand(2, 1, 1, 25),
    target=torch.rand(2, 1, 1, 25),
)


def _bss_eval_images_snr(preds: Tensor, target: Tensor, zero_mean: bool):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            snr_v = mir_eval_bss_eval_images([target[i, j]], [preds[i, j]], compute_permutation=True)[0][0]
            ms.append(snr_v)
        mss.append(ms)
    return torch.tensor(mss)


def _average_metric(preds: Tensor, target: Tensor, metric_func: Callable):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


mireval_snr_zeromean = partial(_bss_eval_images_snr, zero_mean=True)
mireval_snr_nozeromean = partial(_bss_eval_images_snr, zero_mean=False)


@pytest.mark.parametrize(
    "preds, target, ref_metric, zero_mean",
    [
        (inputs.preds, inputs.target, mireval_snr_zeromean, True),
        (inputs.preds, inputs.target, mireval_snr_nozeromean, False),
    ],
)
class TestSNR(MetricTester):
    """Test class for `SignalNoiseRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_snr(self, preds, target, ref_metric, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalNoiseRatio,
            reference_metric=partial(_average_metric, metric_func=ref_metric),
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_functional(self, preds, target, ref_metric, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            signal_noise_ratio,
            ref_metric,
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_snr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=SignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
