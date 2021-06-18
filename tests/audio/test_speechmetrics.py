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
from torch import Tensor


def test_import_speechmetrics() -> None:
    try:
        import speechmetrics
    except ImportError:
        pytest.fail('ImportError speechmetrics')


from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester

seed_all(42)

Time = 100

Input = namedtuple('Input', ["preds", "target"])

inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
)

import multiprocessing
import time

import speechmetrics

speechmetrics_sisdr = speechmetrics.load('sisdr')


def speechmetrics_si_sdr(preds: Tensor, target: Tensor, zero_mean: bool) -> Tensor:
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
            ms.append(metric['sisdr'][0])
        mss.append(ms)
    return torch.tensor(mss)


def test_speechmetrics_si_sdr() -> None:
    t = multiprocessing.Process(target=speechmetrics_si_sdr, args=(inputs.preds[0], inputs.target[0], False))
    t.start()
    try:
        t.join(timeout=180)  # 3min
        if t.is_alive():
            pytest.fail(f'timeout 3min. t.is_alive()={t.is_alive()}')
            t.terminate()
    except:
        pytest.fail('join except')


from torchmetrics.audio import SI_SDR
from torchmetrics.functional import si_sdr

speechmetrics_si_sdr_zero_mean = partial(speechmetrics_si_sdr, zero_mean=True)
speechmetrics_si_sdr_no_zero_mean = partial(speechmetrics_si_sdr, zero_mean=False)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6


# def average_metric(preds, target, metric_func):
#     # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
#     # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
#     return metric_func(preds, target).mean()


@pytest.mark.parametrize(
    "preds, target, sk_metric, zero_mean",
    [
        (inputs.preds, inputs.target, speechmetrics_si_sdr_zero_mean, True),
        (inputs.preds, inputs.target, speechmetrics_si_sdr_no_zero_mean, False),
    ],
)
class TestSISDR(MetricTester):
    atol = 1e-2

    # @pytest.mark.parametrize("ddp", [True, False])
    # @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    # def test_si_sdr(self, preds, target, sk_metric, zero_mean, ddp, dist_sync_on_step):
    #     self.run_class_metric_test(
    #         ddp,
    #         preds,
    #         target,
    #         SI_SDR,
    #         sk_metric=partial(average_metric, metric_func=sk_metric),
    #         dist_sync_on_step=dist_sync_on_step,
    #         metric_args=dict(zero_mean=zero_mean),
    #     )

    def test_si_sdr_functional(self, preds, target, sk_metric, zero_mean):
        self.run_functional_metric_test(
            preds,
            target,
            si_sdr,
            sk_metric,
            metric_args=dict(zero_mean=zero_mean),
        )

    # def test_si_sdr_differentiability(self, preds, target, sk_metric, zero_mean):
    #     self.run_differentiability_test(
    #         preds=preds,
    #         target=target,
    #         metric_module=SI_SDR,
    #         metric_functional=si_sdr,
    #         metric_args={'zero_mean': zero_mean}
    #     )

    # @pytest.mark.skipif(
    #     not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    # )
    # def test_si_sdr_half_cpu(self, preds, target, sk_metric, zero_mean):
    #     pytest.xfail("SI-SDR metric does not support cpu + half precision")

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    # def test_si_sdr_half_gpu(self, preds, target, sk_metric, zero_mean):
    #     self.run_precision_test_gpu(
    #         preds=preds,
    #         target=target,
    #         metric_module=SI_SDR,
    #         metric_functional=si_sdr,
    #         metric_args={'zero_mean': zero_mean}
    #     )


# def test_error_on_different_shape(metric_class=SI_SDR):
#     metric = metric_class()
#     with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
#         metric(torch.randn(100, ), torch.randn(50, ))
