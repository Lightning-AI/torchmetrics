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
from functools import partial

from tests.retrieval.helpers import _test_dtypes, _test_input_shapes, _test_retrieval_against_sklearn
from torchmetrics.retrieval.mean_average_precision import RetrievalMAP
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision

import pytest
from sklearn.metrics import average_precision_score as sk_average_precision_score

from tests.retrieval.inputs import _input_retrieval_scores
from tests.retrieval.helpers import _compute_sklearn_metric

from tests.helpers import seed_all
from tests.helpers.testers import MetricTester

seed_all(42)


@pytest.mark.parametrize(
    "preds, target, idx, sk_metric", [
        (_input_retrieval_scores.preds, _input_retrieval_scores.target, _input_retrieval_scores.idx, sk_average_precision_score),
    ]
)
class TestRetrievalMetric(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("empty_target_action", ['skip', 'neg', 'pos'])
    def test_average_precision(self, preds, target, idx, sk_metric, ddp, dist_sync_on_step, empty_target_action):
        _sk_metric = partial(_compute_sklearn_metric, metric=sk_metric)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=RetrievalMAP,
            sk_metric=_sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={'empty_target_action': empty_target_action},
            idx=idx
        )

    def test_average_precision_functional(self, preds, target, idx, sk_metric):
        _sk_metric = partial(_compute_sklearn_metric, metric=sk_metric, empty_target_action="neg", idx=None)
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=retrieval_average_precision,
            sk_metric=_sk_metric,
        )
