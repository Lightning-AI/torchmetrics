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

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    coverage_error, label_ranking_average_precision_score, label_ranking_loss
)
from torch import tensor, Tensor
from torchmetrics import CoverageError

from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_BATCHES, NUM_CLASSES, THRESHOLD, MetricTester


seed_all(42)


@pytest.mark.parametrize("metric", []
)
@pytest.mark.parametrize(
    "preds, target, subset_accuracy",
    [
        (_input_mlb.preds, _input_mlb.target, False),
        (_input_mlb_logits.preds, _input_mlb_logits.target, False),
        (_input_mlb_prob.preds, _input_mlb_prob.target, False),
    ],
)
class TestRanking(MetricTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_accuracy_class(self, ddp, dist_sync_on_step, preds, target, subset_accuracy):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
        )
"""
    def test_accuracy_fn(self, preds, target, subset_accuracy):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
        )

    def test_accuracy_differentiability(self, preds, target, subset_accuracy):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Accuracy,
            metric_functional=accuracy,
            metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
        )
"""