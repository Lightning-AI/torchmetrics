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
from typing import Tuple

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as _sk_average_precision_score
from sklearn.metrics import precision_recall_curve as _sk_precision_recall_curve
from torch import Tensor

from tests.classification.inputs import _input_binary_prob
from tests.classification.inputs import _input_binary_prob_plausible as _input_binary_prob_ok
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.classification.inputs import _input_multilabel_prob_plausible as _input_mlb_prob_ok
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torchmetrics.classification.binned_precision_recall import BinnedAveragePrecision, BinnedRecallAtFixedPrecision

seed_all(42)


def recall_at_precision_x_multilabel(predictions: Tensor, targets: Tensor, min_precision: float) -> Tuple[float, float]:
    precision, recall, thresholds = _sk_precision_recall_curve(targets, predictions)

    try:
        tuple_all = [(r, p, t) for p, r, t in zip(precision, recall, thresholds) if p >= min_precision]
        max_recall, _, best_threshold = max(tuple_all)
    except ValueError:
        max_recall, best_threshold = 0, 1e6

    return float(max_recall), float(best_threshold)


def _sk_prec_recall_mclass_prob(predictions, targets, num_classes, min_precision):
    max_recalls = torch.zeros(num_classes)
    best_thresholds = torch.zeros(num_classes)

    for i in range(num_classes):
        max_recalls[i], best_thresholds[i] = recall_at_precision_x_multilabel(
            predictions[:, i], targets[:, i], min_precision
        )
    return max_recalls, best_thresholds


def _sk_prec_recall_binary_prob(predictions, targets, num_classes, min_precision):
    return recall_at_precision_x_multilabel(predictions, targets, min_precision)


def _sk_avg_prec_multiclass(predictions, targets, num_classes):
    # replace nan with 0
    return np.nan_to_num(_sk_average_precision_score(targets, predictions, average=None))


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_prec_recall_binary_prob, 1),
        (_input_binary_prob_ok.preds, _input_binary_prob_ok.target, _sk_prec_recall_binary_prob, 1),
        (_input_mlb_prob_ok.preds, _input_mlb_prob_ok.target, _sk_prec_recall_mclass_prob, NUM_CLASSES),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_prec_recall_mclass_prob, NUM_CLASSES),
    ],
)
class TestBinnedRecallAtPrecision(MetricTester):
    atol = 0.02

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("min_precision", [0.05, 0.1, 0.3, 0.5, 0.8, 0.95])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step, min_precision):
        # rounding will simulate binning for both implementations
        preds = Tensor(np.round(preds.numpy(), 2)) + 1e-6

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinnedRecallAtFixedPrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes, min_precision=min_precision),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "min_precision": min_precision,
                "num_thresholds": 101,
            },
        )


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_avg_prec_multiclass, 1),
        (_input_binary_prob_ok.preds, _input_binary_prob_ok.target, _sk_avg_prec_multiclass, 1),
        (_input_mlb_prob_ok.preds, _input_mlb_prob_ok.target, _sk_avg_prec_multiclass, NUM_CLASSES),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_avg_prec_multiclass, NUM_CLASSES),
    ],
)
class TestBinnedAveragePrecision(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("num_thresholds", [101, 301])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step, num_thresholds):
        # rounding will simulate binning for both implementations
        preds = Tensor(np.round(preds.numpy(), 2)) + 1e-6

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinnedAveragePrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "num_thresholds": num_thresholds,
            },
        )
