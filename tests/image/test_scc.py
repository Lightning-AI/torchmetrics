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
from sewar.full_ref import scc

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.image.scc import spatial_correlation_coefficient
from torchmetrics.image.scc import SpatialCorrelationCoefficient

seed_all(42)


def _reference_scc(preds, target, reduction):
    val = 0.0
    for p, t in zip(preds, target):
        val += scc(t.permute(1, 2, 0).numpy(), p.permute(1, 2, 0).numpy(), ws=9)
    val = val if reduction == "sum" else val / preds.shape[0]
    return val


Input = namedtuple("Input", ["preds", "target"])

_inputs = []
for size, channel, dtype in [
    (12, 3, torch.uint8),
    (13, 3, torch.float32),
    (14, 3, torch.double),
    (15, 3, torch.float64),
]:
    preds = torch.randint(0, 255, (NUM_BATCHES, BATCH_SIZE, channel, size, size), dtype=dtype)
    target = torch.randint(0, 255, (NUM_BATCHES, BATCH_SIZE, channel, size, size), dtype=dtype)
    _inputs.append(Input(preds=preds, target=target))


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
class TestSpatialCorrelationCoefficient(MetricTester):
    atol = 1e-3

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_scc(self, reduction, preds, target, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpatialCorrelationCoefficient,
            partial(_reference_scc, reduction=reduction),
            dist_sync_on_step,
            metric_args=dict(reduction=reduction),
        )

    def test_scc_functional(self, reduction, preds, target):
        self.run_functional_metric_test(
            preds,
            target,
            spatial_correlation_coefficient,
            partial(_reference_scc, reduction=reduction),
            metric_args=dict(reduction=reduction),
        )

    # SAM half + cpu does not work due to missing support in torch.log
    @pytest.mark.xfail(reason="SCC metric does not support cpu + half precision")
    def test_scc_half_cpu(self, reduction, preds, target):
        self.run_precision_test_cpu(
            preds,
            target,
            SpatialCorrelationCoefficient,
            spatial_correlation_coefficient,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_scc_half_gpu(self, reduction, preds, target):
        self.run_precision_test_gpu(preds, target, SpatialCorrelationCoefficient, spatial_correlation_coefficient)


def test_error_on_different_shape(metric_class=SpatialCorrelationCoefficient):
    metric = metric_class()
    with pytest.raises(RuntimeError):
        metric(torch.randn([1, 3, 16, 16]), torch.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=SpatialCorrelationCoefficient):
    metric = metric_class()
    with pytest.raises(ValueError):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16]))
