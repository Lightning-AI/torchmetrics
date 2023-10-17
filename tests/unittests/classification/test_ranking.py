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
from functools import partial

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import coverage_error as sk_coverage_error
from sklearn.metrics import label_ranking_average_precision_score as sk_label_ranking
from sklearn.metrics import label_ranking_loss as sk_label_ranking_loss
from torchmetrics.classification.ranking import (
    MultilabelCoverageError,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss,
)
from torchmetrics.functional.classification.ranking import (
    multilabel_coverage_error,
    multilabel_ranking_average_precision,
    multilabel_ranking_loss,
)

from unittests import NUM_CLASSES
from unittests.classification.inputs import _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester, inject_ignore_index

seed_all(42)


def _sklearn_ranking(preds, target, fn, ignore_index):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating) and not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    if ignore_index is not None:
        idx = target == ignore_index
        target[idx] = -1
    return fn(target, preds)


@pytest.mark.parametrize(
    "metric, functional_metric, ref_metric",
    [
        (MultilabelCoverageError, multilabel_coverage_error, sk_coverage_error),
        (MultilabelRankingAveragePrecision, multilabel_ranking_average_precision, sk_label_ranking),
        (MultilabelRankingLoss, multilabel_ranking_loss, sk_label_ranking_loss),
    ],
)
@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelRanking(MetricTester):
    """Test class for `MultilabelRanking` metric."""

    @pytest.mark.parametrize("ignore_index", [None])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_ranking(self, inputs, metric, functional_metric, ref_metric, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            reference_metric=partial(_sklearn_ranking, fn=ref_metric, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None])
    def test_multilabel_ranking_functional(self, inputs, metric, functional_metric, ref_metric, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(_sklearn_ranking, fn=ref_metric, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_ranking_differentiability(self, inputs, metric, functional_metric, ref_metric):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_ranking_dtype_cpu(self, inputs, metric, functional_metric, ref_metric, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        if dtype == torch.half and functional_metric == multilabel_ranking_average_precision:
            pytest.xfail(
                reason="multilabel_ranking_average_precision requires torch.unique which is not implemented for half"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_ranking_dtype_gpu(self, inputs, metric, functional_metric, ref_metric, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
            dtype=dtype,
        )
