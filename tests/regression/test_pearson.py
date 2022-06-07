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

import pytest
import torch
from scipy.stats import pearsonr

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.regression.pearson import PearsonCorrCoef

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_single_target_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_single_target_inputs2 = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE),
)


def _sk_pearsonr(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return pearsonr(sk_target, sk_preds)[0]


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
    ],
)
class TestPearsonCorrcoef(MetricTester):
    @pytest.mark.parametrize("compute_on_cpu", [True, False])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_pearson_corrcoef(self, preds, target, compute_on_cpu, ddp):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=PearsonCorrCoef,
            sk_metric=_sk_pearsonr,
            dist_sync_on_step=False,
            metric_args={"compute_on_cpu": compute_on_cpu},
        )

    def test_pearson_corrcoef_functional(self, preds, target):
        self.run_functional_metric_test(
            preds=preds, target=target, metric_functional=pearson_corrcoef, sk_metric=_sk_pearsonr
        )

    def test_pearson_corrcoef_differentiability(self, preds, target):
        self.run_differentiability_test(
            preds=preds, target=target, metric_module=PearsonCorrCoef, metric_functional=pearson_corrcoef
        )

    # Pearson half + cpu does not work due to missing support in torch.sqrt
    @pytest.mark.xfail(reason="PearsonCorrCoef metric does not support cpu + half precision")
    def test_pearson_corrcoef_half_cpu(self, preds, target):
        self.run_precision_test_cpu(preds, target, PearsonCorrCoef, pearson_corrcoef)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pearson_corrcoef_half_gpu(self, preds, target):
        self.run_precision_test_gpu(preds, target, PearsonCorrCoef, pearson_corrcoef)


def test_error_on_different_shape():
    metric = PearsonCorrCoef()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))

    with pytest.raises(ValueError, match="Expected both predictions and target to be 1 dimensional tensors."):
        metric(torch.randn(100, 2), torch.randn(100, 2))
