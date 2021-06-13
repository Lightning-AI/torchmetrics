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
from asteroid.losses import pairwise_neg_snr
from mir_eval.separation import bss_eval_images

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.audio import SNR
from torchmetrics.functional import snr
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Time = 1000

Input = namedtuple('Input', ["preds", "target"])

inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
)


def asteroid_snr(preds, target):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    snr_v = -pairwise_neg_snr(preds, target)
    return snr_v.view(BATCH_SIZE, 1)


def bss_eval_images_snr(preds, target):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    snr_vb = []
    for j in range(BATCH_SIZE):
        snr_v = bss_eval_images([target[j].view(-1).numpy()], [preds[j].view(-1).numpy()])[0][0][0]
        snr_vb.append(snr_v)
    return torch.tensor(snr_vb)


def average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


@pytest.mark.parametrize(
    "preds, target, sk_metric, zero_mean",
    [
        (inputs.preds, inputs.target, asteroid_snr, True),
        (inputs.preds, inputs.target, bss_eval_images_snr, False),
    ],
)
class TestSNR(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_snr(self, preds, target, sk_metric, zero_mean, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SNR,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_snr_functional(self, preds, target, sk_metric, zero_mean):
        self.run_functional_metric_test(
            preds,
            target,
            snr,
            sk_metric,
            metric_args=dict(zero_mean=zero_mean),
        )

    def test_snr_differentiability(self, preds, target, sk_metric, zero_mean):
        self.run_differentiability_test(
            preds=preds, target=target, metric_module=SNR, metric_functional=snr, metric_args={'zero_mean': zero_mean}
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    )
    def test_snr_half_cpu(self, preds, target, sk_metric, zero_mean):
        self.run_precision_test_cpu(
            preds=preds, target=target, metric_module=SNR, metric_functional=snr, metric_args={'zero_mean': zero_mean}
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    def test_snr_half_gpu(self, preds, target, sk_metric, zero_mean):
        self.run_precision_test_gpu(
            preds=preds, target=target, metric_module=SNR, metric_functional=snr, metric_args={'zero_mean': zero_mean}
        )


def test_error_on_different_shape(metric_class=SNR):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))
