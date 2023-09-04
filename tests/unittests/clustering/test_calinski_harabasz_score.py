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
from sklearn.metrics import calinski_harabasz_score as sklearn_calinski_harabasz_score
from torchmetrics.clustering.calinski_harabasz_score import CalinskiHarabaszScore
from torchmetrics.functional.clustering.calinski_harabasz_score import calinski_harabasz_score

from unittests.clustering.inputs import _single_target_intrinsic1, _single_target_intrinsic2
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_intrinsic1.preds, _single_target_intrinsic1.target),
        (_single_target_intrinsic2.preds, _single_target_intrinsic2.target),
    ],
)
class TestCalinskiHarabaszScore(MetricTester):
    """Test class for `CalinskiHarabaszScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [True, False])
    def test_calinski_harabasz_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CalinskiHarabaszScore,
            reference_metric=sklearn_calinski_harabasz_score,
        )

    def test_calinski_harabasz_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=calinski_harabasz_score,
            reference_metric=sklearn_calinski_harabasz_score,
        )
