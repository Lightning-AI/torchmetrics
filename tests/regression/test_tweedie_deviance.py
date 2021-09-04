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
from sklearn.metrics import mean_tweedie_deviance
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.regression.tweedie_deviance import tweedie_deviance_score
from torchmetrics.regression.tweedie_deviance import TweedieDevianceScore

seed_all(42)

Input = namedtuple("Input", ["preds", "targets"])

_single_target_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    targets=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_single_target_inputs2 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    targets=torch.rand(NUM_BATCHES, BATCH_SIZE),
)


def _sk_deviance(preds: Tensor, targets: Tensor, power: int):
    sk_preds = preds.view(-1).numpy()
    sk_target = targets.view(-1).numpy()
    return mean_tweedie_deviance(sk_target, sk_preds, power=power)


@pytest.mark.parametrize("power", [0, 1, 2])
@pytest.mark.parametrize(
    "preds, targets, sk_metric",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.targets, _sk_deviance),
        (_single_target_inputs2.preds, _single_target_inputs2.targets, _sk_deviance),
    ],
)
class TestDevianceScore(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_deviance_scores_class(self, ddp, dist_sync_on_step, preds, targets, power, sk_metric):
        self.run_class_metric_test(
            ddp,
            preds,
            targets,
            TweedieDevianceScore,
            partial(sk_metric, power=power),
            dist_sync_on_step,
            metric_args=dict(power=power),
        )

    def test_deviance_scores_functional(self, preds, targets, power, sk_metric):
        self.run_functional_metric_test(
            preds,
            targets,
            tweedie_deviance_score,
            partial(sk_metric, power=power),
            metric_args=dict(power=power),
        )


def test_error_on_different_shape(metric_class=TweedieDevianceScore):
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_invalid_inputs(metric_class=TweedieDevianceScore):
    metric = metric_class(power=1)
    with pytest.raises(
        ValueError, match="For power=1, 'preds' has to be strictly positive and targets cannot be negative."
    ):
        metric(torch.tensor([-1.0, 2.0, 3.0]), torch.rand(3))

    with pytest.raises(
        ValueError, match="For power=1, 'preds' has to be strictly positive and targets cannot be negative."
    ):
        metric(torch.rand(3), torch.tensor([-1.0, 2.0, 3.0]))

    metric = metric_class(power=2)
    with pytest.raises(ValueError, match="For power=2, both 'preds' and 'targets' have to be strictly positive."):
        metric(torch.tensor([-1.0, 2.0, 3.0]), torch.rand(3))

    with pytest.raises(ValueError, match="For power=2, both 'preds' and 'targets' have to be strictly positive."):
        metric(torch.rand(3), torch.tensor([-1.0, 2.0, 3.0]))
