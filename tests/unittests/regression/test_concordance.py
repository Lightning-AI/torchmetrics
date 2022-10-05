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

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr

from torchmetrics.functional.regression.concordance import concordance_corrcoef
from torchmetrics.regression.concordance import ConcordanceCorrCoef
from unittests.helpers import seed_all
from unittests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, MetricTester

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

_multi_target_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_multi_target_inputs2 = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)


def _sk_concordance(preds, target):
    preds, target = preds.numpy(), target.numpy()
    if preds.ndim == 2:
        mean_pred = np.mean(preds, axis=0)
        mean_gt = np.mean(target, axis=0)
        std_pred = np.std(preds, axis=0)
        std_gt = np.std(target, axis=0)
        pearson = np.stack([pearsonr(t, p)[0] for t, p in zip(target.T, preds.T)])
    else:
        mean_pred = np.mean(preds)
        mean_gt = np.mean(target)
        std_pred = np.std(preds)
        std_gt = np.std(target)
        pearson = pearsonr(target, preds)[0]
    return 2.0 * pearson * std_pred * std_gt / (std_pred**2 + std_gt**2 + (mean_pred - mean_gt) ** 2)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_multi_target_inputs1.preds, _multi_target_inputs1.target),
        (_multi_target_inputs2.preds, _multi_target_inputs2.target),
    ],
)
class TestConcordanceCorrCoef(MetricTester):
    atol = 1e-3

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_concordance_corrcoef(self, preds, target, ddp, dist_sync_on_step):
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ConcordanceCorrCoef,
            _sk_concordance,
            dist_sync_on_step,
            metric_args={"num_outputs": num_outputs},
        )

    def test_concordance_corrcoef_functional(self, preds, target):
        self.run_functional_metric_test(preds, target, concordance_corrcoef, _sk_concordance)

    def test_concordance_corrcoef_differentiability(self, preds, target):
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(ConcordanceCorrCoef, num_outputs=num_outputs),
            metric_functional=concordance_corrcoef,
        )

    # Spearman half + cpu does not work due to missing support in torch.arange
    @pytest.mark.xfail(reason="Concordance metric does not support cpu + half precision")
    def test_concordance_corrcoef_half_cpu(self, preds, target):
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_cpu(
            preds, target, partial(ConcordanceCorrCoef, num_outputs=num_outputs), concordance_corrcoef
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_concordance_corrcoef_half_gpu(self, preds, target):
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_gpu(
            preds, target, partial(ConcordanceCorrCoef, num_outputs=num_outputs), concordance_corrcoef
        )


def test_error_on_different_shape():
    metric = ConcordanceCorrCoef(num_outputs=1)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))

    metric = ConcordanceCorrCoef(num_outputs=5)
    with pytest.raises(ValueError, match="Expected both predictions and target to be either 1- or 2-.*"):
        metric(torch.randn(100, 2, 5), torch.randn(100, 2, 5))

    metric = ConcordanceCorrCoef(num_outputs=2)
    with pytest.raises(ValueError, match="Expected argument `num_outputs` to match the second dimension of input.*"):
        metric(torch.randn(100, 5), torch.randn(100, 5))
