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

import properscoring as ps
import pytest
import torch

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.regression.crps import crps
from torchmetrics.regression.crps import CRPS

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

@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
class TestCosineSimilarity(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_cosine_similarity(self, reduction, preds, target, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            CosineSimilarity,
            partial(sk_metric, reduction=reduction),
            dist_sync_on_step,
            metric_args=dict(reduction=reduction),
        )

    def test_cosine_similarity_functional(self, reduction, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            cosine_similarity,
            partial(sk_metric, reduction=reduction),
            metric_args=dict(reduction=reduction),
        )


def test_error_on_different_shape(metric_class=CosineSimilarity):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))
