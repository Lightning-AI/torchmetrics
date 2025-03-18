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
from sklearn.metrics import davies_bouldin_score as sklearn_davies_bouldin_score

from torchmetrics.clustering.davies_bouldin_score import DaviesBouldinScore
from torchmetrics.functional.clustering.davies_bouldin_score import davies_bouldin_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _single_target_intrinsic1, _single_target_intrinsic2

seed_all(42)


@pytest.mark.parametrize(
    "data, labels",
    [
        (_single_target_intrinsic1.data, _single_target_intrinsic1.labels),
        (_single_target_intrinsic2.data, _single_target_intrinsic2.labels),
    ],
)
class TestDaviesBouldinScore(MetricTester):
    """Test class for `DaviesBouldinScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_davies_bouldin_score(self, data, labels, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=data,
            target=labels,
            metric_class=DaviesBouldinScore,
            reference_metric=sklearn_davies_bouldin_score,
        )

    def test_davies_bouldin_score_functional(self, data, labels):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=data,
            target=labels,
            metric_functional=davies_bouldin_score,
            reference_metric=sklearn_davies_bouldin_score,
        )
