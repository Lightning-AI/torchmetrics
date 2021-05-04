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
from sklearn.metrics import r2_score as sk_r2score

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional import r2score
from torchmetrics.regression import R2Score
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


def _single_target_sk_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        r2_score = 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


def _multi_target_sk_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        r2_score = 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


@pytest.mark.parametrize("adjusted", [0, 5, 10])
@pytest.mark.parametrize("multioutput", ['raw_values', 'uniform_average', 'variance_weighted'])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_outputs",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric, 1),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric, num_targets),
    ],
)
class TestR2Score(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_r2(self, adjusted, multioutput, preds, target, sk_metric, num_outputs, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            R2Score,
            partial(sk_metric, adjusted=adjusted, multioutput=multioutput),
            dist_sync_on_step,
            metric_args=dict(adjusted=adjusted, multioutput=multioutput, num_outputs=num_outputs),
        )

    def test_r2_functional(self, adjusted, multioutput, preds, target, sk_metric, num_outputs):
        # todo: `num_outputs` is unused
        self.run_functional_metric_test(
            preds,
            target,
            r2score,
            partial(sk_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args=dict(adjusted=adjusted, multioutput=multioutput),
        )

    def test_r2_differentiabilit(self, adjusted, multioutput, preds, target, sk_metric, num_outputs):
        self.run_differentiability_test(
            preds=preds, target=target, metric_module=partial(R2Score, num_outputs=num_outputs),
            metric_functional=r2score, metric_args=dict(adjusted=adjusted, multioutput=multioutput)
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    )
    def test_r2_half_cpu(self, adjusted, multioutput, preds, target, sk_metric, num_outputs):
        self.run_precision_test_cpu(
            preds, target, partial(R2Score, num_outputs=num_outputs), r2score, {
                'adjusted': adjusted,
                'multioutput': multioutput
            }
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    def test_r2_half_gpu(self, adjusted, multioutput, preds, target, sk_metric, num_outputs):
        self.run_precision_test_gpu(
            preds, target, partial(R2Score, num_outputs=num_outputs), r2score, {
                'adjusted': adjusted,
                'multioutput': multioutput
            }
        )


def test_error_on_different_shape(metric_class=R2Score):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))


def test_error_on_multidim_tensors(metric_class=R2Score):
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match=r'Expected both prediction and target to be 1D or 2D tensors,'
        r' but received tensors with dimension .'
    ):
        metric(torch.randn(10, 20, 5), torch.randn(10, 20, 5))


def test_error_on_too_few_samples(metric_class=R2Score):
    metric = metric_class()
    with pytest.raises(ValueError, match='Needs at least two samples to calculate r2 score.'):
        metric(torch.randn(1, ), torch.randn(1, ))


def test_warning_on_too_large_adjusted(metric_class=R2Score):
    metric = metric_class(adjusted=10)

    with pytest.warns(
        UserWarning,
        match="More independent regressions than data points in"
        " adjusted r2 score. Falls back to standard r2 score."
    ):
        metric(torch.randn(10, ), torch.randn(10, ))

    with pytest.warns(UserWarning, match="Division by zero in adjusted r2 score. Falls back to" " standard r2 score."):
        metric(torch.randn(11, ), torch.randn(11, ))
