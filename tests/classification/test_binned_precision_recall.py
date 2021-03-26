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

from torchmetrics.functional import precision_recall_curve
from functools import partial
from typing import Tuple

import pytest
import torch
from sklearn.metrics import average_precision_score as _sk_average_precision_score
from torchmetrics.classification.binned_precision_recall import (
    BinnedAveragePrecision,
    BinnedRecallAtFixedPrecision,
)
from tests.classification.inputs import (
    Input,
)
from tests.helpers.testers import (
    NUM_CLASSES,
    NUM_BATCHES,
    BATCH_SIZE,
    MetricTester,
)


torch.manual_seed(42)


def construct_not_terrible_input():
    correct_targets = torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
    targets = torch.zeros_like(preds, dtype=torch.long)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            targets[i, j, correct_targets[i, j]] = 1
    preds += torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES) * targets / 3

    preds = preds / preds.sum(dim=2, keepdim=True)

    return Input(preds=preds, target=targets)


__test_input = construct_not_terrible_input()


def recall_at_precision_x_multilabel(
    precision: torch.Tensor, recall, thresholds: torch.Tensor, min_precision: float
) -> Tuple[float, float]:
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
        precisions, recalls, thresholds = precision_recall_curve(
            predictions[:, i], targets[:, i], pos_label=1
        )
        max_recalls[i], best_thresholds[i] = recall_at_precision_x_multilabel(
            precisions, recalls, thresholds, min_precision
        )
    return max_recalls, best_thresholds


def _multiclass_average_precision_sk_metric(predictions, targets, num_classes):
    return _sk_average_precision_score(targets, predictions, average=None)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (
            __test_input.preds,
            __test_input.target,
            _multiclass_prob_sk_metric,
            NUM_CLASSES,
        ),
    ],
)
class TestBinnedRecallAtPrecision(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp):
        self.atol = 0.05  # up to second decimal using 500 thresholds
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinnedRecallAtFixedPrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes, min_precision=0.6),
            dist_sync_on_step=False,
            check_dist_sync_on_step=False,
            check_batch=False,
            metric_args={
                "num_classes": num_classes,
                "min_precision": 0.6,
                "num_thresholds": 2000,
            },
        )


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (
            __test_input.preds,
            __test_input.target,
            _multiclass_average_precision_sk_metric,
            NUM_CLASSES,
        ),
    ],
)
class TestBinnedAveragePrecision(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binned_pr(self, preds, target, sk_metric, num_classes, ddp):
        self.atol = 0.01  # up to second decimal using 200 thresholds
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
                "num_thresholds": 200,
            },
        )
