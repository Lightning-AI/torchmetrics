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
from sklearn.metrics import explained_variance_score

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional import explained_variance
from torchmetrics.regression import ExplainedVariance
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

num_targets = 5

Input = namedtuple('Input', ["preds", "target"])

_single_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _single_target_sk_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_fn(sk_target, sk_preds)


def _multi_target_sk_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    return sk_fn(sk_target, sk_preds)


@pytest.mark.parametrize("multioutput", ['raw_values', 'uniform_average', 'variance_weighted'])
@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
class TestExplainedVariance(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_explained_variance(self, multioutput, preds, target, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ExplainedVariance,
            partial(sk_metric, sk_fn=partial(explained_variance_score, multioutput=multioutput)),
            dist_sync_on_step,
            metric_args=dict(multioutput=multioutput),
        )

    def test_explained_variance_functional(self, multioutput, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            explained_variance,
            partial(sk_metric, sk_fn=partial(explained_variance_score, multioutput=multioutput)),
            metric_args=dict(multioutput=multioutput),
        )

    def test_explained_variance_differentiability(self, multioutput, preds, target, sk_metric):
        self.run_differentiability_test(
            preds=preds, target=target, metric_module=ExplainedVariance, metric_functional=explained_variance,
            metric_args={'multioutput': multioutput}
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    )
    def test_explained_variance_half_cpu(self, multioutput, preds, target, sk_metric):
        self.run_precision_test_cpu(preds, target, ExplainedVariance, explained_variance)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    def test_explained_variance_half_gpu(self, multioutput, preds, target, sk_metric):
        self.run_precision_test_gpu(preds, target, ExplainedVariance, explained_variance)


def test_error_on_different_shape(metric_class=ExplainedVariance):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))
