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
import pytest
import torch
from sklearn.metrics import rand_score as sklearn_rand_score

from torchmetrics.clustering.rand_score import RandScore
from torchmetrics.functional.clustering.rand_score import rand_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _float_inputs_extrinsic, _single_target_extrinsic1, _single_target_extrinsic2

seed_all(42)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestRandScore(MetricTester):
    """Test class for `RandScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_rand_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=RandScore,
            reference_metric=sklearn_rand_score,
        )

    def test_rand_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=rand_score,
            reference_metric=sklearn_rand_score,
        )


def test_rand_score_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match=r"Expected *"):
        rand_score(preds, target)


def test_rand_score_functional_is_symmetric(
    preds=_single_target_extrinsic1.preds, target=_single_target_extrinsic1.target
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert torch.allclose(rand_score(p, t), rand_score(t, p))
