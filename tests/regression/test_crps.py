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
from properscoring import crps_ensemble

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, MetricTester
from torchmetrics.functional.regression.crps import crps
from torchmetrics.regression.crps import CRPS

seed_all(42)

num_ensemble = 5

Input = namedtuple('Input', ["preds", "target"])

_single_ensample_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_ensample_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_ensemble),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_extra_dim_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_ensemble, EXTRA_DIM),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
)


def _compare_fn(preds, target):
    n_ensemble_members = preds.shape[1]
    ensemble_sum_scale_factor = (1 / (n_ensemble_members * (n_ensemble_members - 1))) if n_ensemble_members > 1 else 1.0
    return ensemble_sum_scale_factor * crps_ensemble(target, preds, axis=1).mean()


@pytest.mark.parametrize(
    "preds, target",
    [(_single_ensample_inputs.preds, _single_ensample_inputs.target),
     (_multi_ensample_inputs.preds, _multi_ensample_inputs.target),
     (_extra_dim_inputs.preds, _extra_dim_inputs.target)],
)
class TestCSPR(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_cspr_module(self, preds, target, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            CRPS,
            _compare_fn,
            dist_sync_on_step,
        )

    def test_cspr_functional(self, preds, target):
        self.run_functional_metric_test(preds, target, crps, _compare_fn)


@pytest.mark.parametrize("preds_shape, target_shape", [((10, 3, 5), (10, )), ((10, 3), (5, ))])
def test_error_on_different_shape(preds_shape, target_shape):
    metric = CRPS()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(*preds_shape), torch.randn(*target_shape))
