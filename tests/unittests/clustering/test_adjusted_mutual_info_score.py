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
from functools import partial

import pytest
import torch
from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami

from torchmetrics.clustering.adjusted_mutual_info_score import AdjustedMutualInfoScore
from torchmetrics.functional.clustering.adjusted_mutual_info_score import adjusted_mutual_info_score
from unittests import BATCH_SIZE, NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _float_inputs_extrinsic, _single_target_extrinsic1, _single_target_extrinsic2

seed_all(42)

ATOL = 1e-5


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
@pytest.mark.parametrize(
    "average_method",
    ["min", "arithmetic", "geometric", "max"],
)
class TestAdjustedMutualInfoScore(MetricTester):
    """Test class for `AdjustedMutualInfoScore` metric."""

    atol = ATOL

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_adjusted_mutual_info_score(self, preds, target, average_method, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AdjustedMutualInfoScore,
            reference_metric=partial(sklearn_ami, average_method=average_method),
            metric_args={"average_method": average_method},
        )

    def test_adjusted_mutual_info_score_functional(self, preds, target, average_method):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=adjusted_mutual_info_score,
            reference_metric=partial(sklearn_ami, average_method=average_method),
            average_method=average_method,
        )


@pytest.mark.parametrize("average_method", ["min", "geometric", "arithmetic", "max"])
def test_adjusted_mutual_info_score_functional_single_cluster(average_method):
    """Check that for single cluster the metric returns 0."""
    tensor_a = torch.randint(NUM_CLASSES, (BATCH_SIZE,))
    tensor_b = torch.zeros((BATCH_SIZE,), dtype=torch.int)
    assert torch.allclose(adjusted_mutual_info_score(tensor_a, tensor_b, average_method), torch.tensor(0.0), atol=ATOL)
    assert torch.allclose(adjusted_mutual_info_score(tensor_b, tensor_a, average_method), torch.tensor(0.0), atol=ATOL)


@pytest.mark.parametrize("average_method", ["min", "geometric", "arithmetic", "max"])
def test_adjusted_mutual_info_score_functional_raises_invalid_task(average_method):
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match=r"Expected *"):
        adjusted_mutual_info_score(preds, target, average_method)


@pytest.mark.parametrize(
    "average_method",
    ["min", "geometric", "arithmetic", "max"],
)
def test_adjusted_mutual_info_score_functional_is_symmetric(
    average_method, preds=_single_target_extrinsic1.preds, target=_single_target_extrinsic1.target
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert torch.allclose(
            adjusted_mutual_info_score(p, t, average_method),
            adjusted_mutual_info_score(t, p, average_method),
            atol=1e-6,
        )
