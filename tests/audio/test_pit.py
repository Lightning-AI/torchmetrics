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

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.audio import PIT
from torchmetrics.functional import pit, si_sdr, snr
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Time = 10

Input = namedtuple('Input', ["preds", "target"])

inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, Time),
)
inputs2 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, Time),
)


def scipy_version(preds: Tensor, target: Tensor, metric_func: Callable, eval_func: str):
    batch_size, spk_num = target.shape[0:2]
    metric_mtx = torch.empty((batch_size, spk_num, spk_num), device=target.device)
    for t in range(spk_num):
        for e in range(spk_num):
            metric_mtx[:, t, e] = metric_func(preds[:, e, ...], target[:, t, ...])

    # pit_r = PIT(metric_func, eval_func)(preds, target)
    metric_mtx = metric_mtx.detach().cpu().numpy()
    best_metrics = []
    best_perms = []
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(metric_mtx[b, ...], eval_func == 'max')
        best_metrics.append(metric_mtx[b, row_idx, col_idx].mean())
        best_perms.append(col_idx)
    return torch.from_numpy(np.stack(best_metrics)), torch.from_numpy(np.stack(best_perms))


def average_metric(preds: Tensor, target: Tensor, metric_func: Callable):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target)[0].mean()


snr_pit_scipy = partial(scipy_version, metric_func=snr, eval_func='max')
si_sdr_pit_scipy = partial(scipy_version, metric_func=si_sdr, eval_func='max')


@pytest.mark.parametrize(
    "preds, target, sk_metric, metric_func, eval_func",
    [
        (inputs1.preds, inputs1.target, snr_pit_scipy, snr, 'max'),
        (inputs1.preds, inputs1.target, si_sdr_pit_scipy, si_sdr, 'max'),
        (inputs2.preds, inputs2.target, snr_pit_scipy, snr, 'max'),
        (inputs2.preds, inputs2.target, si_sdr_pit_scipy, si_sdr, 'max'),
    ],
)
class TestPIT(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_pit(self, preds, target, sk_metric, metric_func, eval_func, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PIT,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(metric_func=metric_func, eval_func=eval_func),
        )

    def test_pit_functional(self, preds, target, sk_metric, metric_func, eval_func):
        device = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'

        # move to device
        preds = preds.to(device)
        target = target.to(device)

        for i in range(NUM_BATCHES):
            best_metric, best_perm = pit(preds[i], target[i], metric_func, eval_func)
            best_metric_sk, best_perm_sk = sk_metric(preds[i].cpu(), target[i].cpu())

            # assert its the same
            assert np.allclose(
                best_metric.detach().cpu().numpy(), best_metric_sk.detach().cpu().numpy(), atol=self.atol
            )
            assert (best_perm.detach().cpu().numpy() == best_perm_sk.detach().cpu().numpy()).all()

    def test_pit_differentiability(self, preds, target, sk_metric, metric_func, eval_func):

        def pit_diff(preds, target, metric_func, eval_func):
            return pit(preds, target, metric_func, eval_func)[0]

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PIT,
            metric_functional=pit_diff,
            metric_args={
                'metric_func': metric_func,
                'eval_func': eval_func
            }
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    )
    def test_pit_half_cpu(self, preds, target, sk_metric, metric_func, eval_func):
        pytest.xfail("PIT metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    def test_pit_half_gpu(self, preds, target, sk_metric, metric_func, eval_func):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PIT,
            metric_functional=partial(pit, metric_func=metric_func, eval_func=eval_func),
            metric_args={
                'metric_func': metric_func,
                'eval_func': eval_func
            }
        )


def test_error_on_different_shape() -> None:
    metric = PIT(snr, 'max')
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(3, 3, 10), torch.randn(3, 2, 10))


def test_error_on_wrong_eval_func() -> None:
    metric = PIT(snr, 'xxx')
    with pytest.raises(RuntimeError, match='eval_func can only be "max" or "min"'):
        metric(torch.randn(3, 3, 10), torch.randn(3, 3, 10))


def test_error_on_wrong_shape() -> None:
    metric = PIT(snr, 'max')
    with pytest.raises(RuntimeError, match='Inputs must be of shape *'):
        metric(torch.randn(3), torch.randn(3))
