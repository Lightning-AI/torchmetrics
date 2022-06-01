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

from tests.helpers import seed_all
from tests.helpers.reference_metrics import _sk_ergas
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis

seed_all(42)

Input = namedtuple("Input", ["preds", "target", "ratio"])

_inputs = []
for size, channel, coef, ratio, dtype in [
    (12, 1, 0.9, 1, torch.float),
    (13, 3, 0.8, 2, torch.float32),
    (14, 1, 0.7, 3, torch.double),
    (15, 3, 0.5, 4, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(Input(preds=preds, target=preds * coef, ratio=ratio))


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(
    "preds, target, ratio",
    [(i.preds, i.target, i.ratio) for i in _inputs],
)
class TestErrorRelativeGlobalDimensionlessSynthesis(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_ergas(self, reduction, preds, target, ratio, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ErrorRelativeGlobalDimensionlessSynthesis,
            partial(_sk_ergas, ratio=ratio, reduction=reduction),
            dist_sync_on_step,
            metric_args=dict(ratio=ratio, reduction=reduction),
        )

    def test_ergas_functional(self, reduction, preds, target, ratio):
        self.run_functional_metric_test(
            preds,
            target,
            error_relative_global_dimensionless_synthesis,
            partial(_sk_ergas, ratio=ratio, reduction=reduction),
            metric_args=dict(ratio=ratio, reduction=reduction),
        )

    # ERGAS half + cpu does not work due to missing support in torch.log
    @pytest.mark.xfail(reason="ERGAS metric does not support cpu + half precision")
    def test_ergas_half_cpu(self, reduction, preds, target, ratio):
        self.run_precision_test_cpu(
            preds,
            target,
            ErrorRelativeGlobalDimensionlessSynthesis,
            error_relative_global_dimensionless_synthesis,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_ergas_half_gpu(self, reduction, preds, target, ratio):
        self.run_precision_test_gpu(
            preds, target, ErrorRelativeGlobalDimensionlessSynthesis, error_relative_global_dimensionless_synthesis
        )


def test_error_on_different_shape(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    metric = metric_class()
    with pytest.raises(RuntimeError):
        metric(torch.randn([1, 3, 16, 16]), torch.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    metric = metric_class()
    with pytest.raises(ValueError):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16]))


def test_error_on_invalid_type(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    metric = metric_class()
    with pytest.raises(TypeError):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16], dtype=torch.float64))
