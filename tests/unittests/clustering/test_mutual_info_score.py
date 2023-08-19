# Copyright The Lightning team.
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
from sklearn.metrics import mutual_info_score as scipy_mutual_info_score
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.clustering.mutual_info_score import MutualInfoScore

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])
NUM_CLASSES = 10

_single_target_inputs1 = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

_single_target_inputs2 = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

_float_inputs = Input(
    preds=torch.rand((NUM_BATCHES, BATCH_SIZE)),
    target=torch.rand((NUM_BATCHES, BATCH_SIZE)),
)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
    ],
)
class TestMutualInfoScore(MetricTester):
    """Test class for `MutualInfoScore` metric."""

    atol = 1e-3

    @pytest.mark.parametrize("compute_on_cpu", [True, False])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_mutual_info_score(self, preds, target, compute_on_cpu, ddp):
        """Test class implementation of metric."""
        metric_args = {"num_classes": NUM_CLASSES}
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MutualInfoScore,
            reference_metric=scipy_mutual_info_score,
            metric_args=metric_args
        )

    def test_mutual_info_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=mutual_info_score,
            reference_metric=scipy_mutual_info_score
        )


def test_mutual_info_score_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs
    with pytest.raises(ValueError, match=r"Expected discrete *"):
        mutual_info_score(preds, target)
