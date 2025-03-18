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
from sklearn.metrics import fowlkes_mallows_score as sklearn_fowlkes_mallows_score

from torchmetrics.clustering import FowlkesMallowsIndex
from torchmetrics.functional.clustering import fowlkes_mallows_index
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _single_target_extrinsic1, _single_target_extrinsic2

seed_all(42)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestFowlkesMallowsIndex(MetricTester):
    """Test class for `FowlkesMallowsIndex` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fowlkes_mallows_index(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=FowlkesMallowsIndex,
            reference_metric=sklearn_fowlkes_mallows_score,
        )

    def test_fowlkes_mallows_index_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=fowlkes_mallows_index,
            reference_metric=sklearn_fowlkes_mallows_score,
        )
