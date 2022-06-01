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
from typing import Callable

import pytest
import torch
from mir_eval.separation import bss_eval_images as mir_eval_bss_eval_images
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import NUM_BATCHES, MetricTester
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.functional import signal_noise_ratio
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

TIME = 100

Input = namedtuple("Input", ["preds", "target"])

BATCH_SIZE = 2
inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, TIME),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, TIME),
)


def bss_eval_images_snr(preds: Tensor, target: Tensor, metric_func: Callable, zero_mean: bool):
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
            if metric_func == mir_eval_bss_eval_images:
                snr_v = metric_func([target[i, j]], [preds[i, j]])[0][0]
            else:
                snr_v = metric_func([target[i, j]], [preds[i, j]])[0][0][0]
            ms.append(snr_v)
        mss.append(ms)
    return torch.tensor(mss)


def average_metric(preds: Tensor, target: Tensor, metric_func: Callable):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


mireval_snr_zeromean = partial(bss_eval_images_snr, metric_func=mir_eval_bss_eval_images, zero_mean=True)
mireval_snr_nozeromean = partial(bss_eval_images_snr, metric_func=mir_eval_bss_eval_images, zero_mean=False)


@pytest.mark.parametrize(
    "preds, target, sk_metric, zero_mean",
    [
        (inputs.preds, inputs.target, mireval_snr_zeromean, True),
        (inputs.preds, inputs.target, mireval_snr_nozeromean, False),
    ],
)
class TestSNR(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_snr(self, preds, target, sk_metric, zero_mean, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalNoiseRatio,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_snr_functional(self, preds, target, sk_metric, zero_mean):
        self.run_functional_metric_test(
            preds,
            target,
            signal_noise_ratio,
            sk_metric,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_snr_differentiability(self, preds, target, sk_metric, zero_mean):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_snr_half_cpu(self, preds, target, sk_metric, zero_mean):
        pytest.xfail("SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_snr_half_gpu(self, preds, target, sk_metric, zero_mean):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=SignalNoiseRatio):
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
