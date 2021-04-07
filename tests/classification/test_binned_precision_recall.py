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

import pytest
import torch
from sklearn.metrics import average_precision_score as _sk_average_precision_score

from tests.classification.inputs import (
    _input_binary_prob,
    _input_binary_prob_plausible,
    _input_multilabel_prob,
    _input_multilabel_prob_plausible,
)
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torchmetrics.classification.binned_precision_recall import BinnedAveragePrecision, BinnedRecallAtFixedPrecision
from torchmetrics.functional import precision_recall_curve

seed_all(42)


def recall_at_precision_x_multilabel(
    predictions: torch.Tensor, targets: torch.Tensor, min_precision: float
) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        predictions, targets, pos_label=1
    )

    try:
        max_recall, max_precision, best_threshold = max(
            (r, p, t)
            for p, r, t in zip(precision, recall, thresholds)
            if p >= min_precision
        )
    except ValueError:
        max_recall, best_threshold = 0, 1e6

    return max_recall, best_threshold


def _multiclass_prob_sk_metric(predictions, targets, num_classes, min_precision):
    max_recalls = torch.zeros(num_classes)
    best_thresholds = torch.zeros(num_classes)

    for i in range(num_classes):
        max_recalls[i], best_thresholds[i] = recall_at_precision_x_multilabel(
            predictions[:, i], targets[:, i], min_precision
        )
    return max_recalls, best_thresholds


def _binary_prob_sk_metric(predictions, targets, num_classes, min_precision):
    return recall_at_precision_x_multilabel(
        predictions, targets, min_precision
    )


def _multiclass_average_precision_sk_metric(predictions, targets, num_classes):
    return _sk_average_precision_score(targets, predictions, average=None)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _binary_prob_sk_metric, 1),
        (_input_binary_prob_plausible.preds, _input_binary_prob_plausible.target, _binary_prob_sk_metric, 1),
        (
            _input_multilabel_prob_plausible.preds,
            _input_multilabel_prob_plausible.target,
            _multiclass_prob_sk_metric,
            NUM_CLASSES,
        ),
        (
            _input_multilabel_prob.preds,
            _input_multilabel_prob.target,
            _multiclass_prob_sk_metric,
            NUM_CLASSES,
        ),
    ],
)
class TestBinnedRecallAtPrecision(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("min_precision", [0.1, 0.3, 0.5])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp, min_precision):
        self.atol = 0.05  # Binned and SKLearn implementations can produce different values
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinnedRecallAtFixedPrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes, min_precision=min_precision),
            dist_sync_on_step=False,
            check_dist_sync_on_step=False,
            check_batch=False,
            metric_args={
                "num_classes": num_classes,
                "min_precision": min_precision,
                "num_thresholds": 2000,
            },
        )


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _multiclass_average_precision_sk_metric, 1),
        (_input_binary_prob_plausible.preds, _input_binary_prob_plausible.target, _multiclass_average_precision_sk_metric, 1),
        (
            _input_multilabel_prob_plausible.preds,
            _input_multilabel_prob_plausible.target,
            _multiclass_average_precision_sk_metric,
            NUM_CLASSES,
        ),
        (
            _input_multilabel_prob.preds,
            _input_multilabel_prob.target,
            _multiclass_average_precision_sk_metric,
            NUM_CLASSES,
        ),
    ],
)
class TestBinnedAveragePrecision(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("num_thresholds", [200, 300])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp, num_thresholds):
        self.atol = 0.01
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinnedAveragePrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=False,
            check_dist_sync_on_step=False,
            check_batch=False,
            metric_args={
                "num_classes": num_classes,
                "num_thresholds": num_thresholds,
            },
        )
