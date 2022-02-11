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
import pytest
from sklearn.metrics import coverage_error as sk_coverage_error

from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import MetricTester
from torchmetrics.classification.ranking import CoverageError
from torchmetrics.functional.classification.ranking import coverage_error

# from sklearn.metrics import label_ranking_average_precision_score as sk_label_ranking
# from sklearn.metrics import label_ranking_loss as sk_label_ranking_loss


seed_all(42)


def _sk_coverage_error(preds, target):
    return sk_coverage_error(target, preds)


@pytest.mark.parametrize("metric, functional_metric, sk_metric", [(CoverageError, coverage_error, _sk_coverage_error)])
@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_mlb.preds, _input_mlb.target),
        (_input_mlb_logits.preds, _input_mlb_logits.target),
        (_input_mlb_prob.preds, _input_mlb_prob.target),
    ],
)
class TestRanking(MetricTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_ranking_class(self, ddp, dist_sync_on_step, preds, target, metric, functional_metric, sk_metric):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_ranking_functional(self, preds, target, metric, functional_metric, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=functional_metric,
            sk_metric=sk_metric,
        )

    def test_ranking_differentiability(self, preds, target, metric, functional_metric, sk_metric):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
        )
