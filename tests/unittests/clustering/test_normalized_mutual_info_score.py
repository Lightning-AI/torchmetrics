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

import pytest
import torch
from sklearn.metrics import normalized_mutual_info_score as sklearn_nmi
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.functional.clustering import normalized_mutual_info_score

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
@pytest.mark.parametrize(
    "normalization_method", ["min", "arithmetic", "geometric", "max"],
)
class TestNormalizedMutualInfoScore(MetricTester):
    """Test class for `NormalizedMutualInfoScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [True, False])
    def test_normalized_mutual_info_score(self, preds, target, normalization_method, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=NormalizedMutualInfoScore,
            reference_metric=sklearn_nmi,
            normalization_method=normalization_method,
        )

    def test_normalized_mutual_info_score_functional(self, preds, target, normalization_method):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=normalized_mutual_info_score,
            reference_metric=sklearn_nmi,
            normalization_method=normalization_method,
        )


@pytest.mark.parametrize("normalization_method", ["min", "geometric", "arithmetic", "max"])
def test_normalized_mutual_info_score_functional_single_cluster(normalization_method):
    """Check that for single cluster the metric returns 0."""
    tensor_a = torch.randint(NUM_CLASSES, (BATCH_SIZE,))
    tensor_b = torch.zeros(BATCH_SIZE, dtype=torch.int)
    assert torch.allclose(normalized_mutual_info_score(tensor_a, tensor_b, normalization_method), torch.tensor(0.0))
    assert torch.allclose(normalized_mutual_info_score(tensor_b, tensor_a, normalization_method), torch.tensor(0.0))


@pytest.mark.parametrize("normalization_method", ["min", "geometric", "arithmetic", "max"])
def test_normalized_mutual_info_score_functional_raises_invalid_task(normalization_method):
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs
    with pytest.raises(ValueError, match=r"Expected *"):
        normalized_mutual_info_score(preds, target, normalization_method)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
    ],
)
@pytest.mark.parametrize(
    "normalization_method",
    ["min", "geometric", "arithmetic", "max"],
)
def test_normalized_mutual_info_score_functional_is_symmetric(preds, target, normalization_method):
    """Check that the metric funtional is symmetric."""
    for p, t in zip(preds, target):
        assert torch.allclose(
            normalized_mutual_info_score(p, t, normalization_method),
            normalized_mutual_info_score(t, p, normalization_method),
        )
