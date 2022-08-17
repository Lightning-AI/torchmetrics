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
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6, _TORCH_GREATER_EQUAL_1_9
from unittests.classification.inputs import _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index

seed_all(42)


def _sk_ranking(preds, target, fn, ignore_index):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    if ignore_index is not None:
        idx = target == ignore_index
        target[idx] = -1
    return fn(target, preds)


@pytest.mark.parametrize(
    "metric, functional_metric, sk_metric",
    [
        (MultilabelCoverageError, multilabel_coverage_error, sk_coverage_error),
        (MultilabelRankingAveragePrecision, multilabel_ranking_average_precision, sk_label_ranking),
        (MultilabelRankingLoss, multilabel_ranking_loss, sk_label_ranking_loss),
    ],
)
@pytest.mark.parametrize(
    "input", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelRanking(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_ranking(self, input, metric, functional_metric, sk_metric, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            sk_metric=partial(_sk_ranking, fn=sk_metric, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None])
    def test_multilabel_ranking_functional(self, input, metric, functional_metric, sk_metric, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            sk_metric=partial(_sk_ranking, fn=sk_metric, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_ranking_differentiability(self, input, metric, functional_metric, sk_metric):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_ranking_dtype_cpu(self, input, metric, functional_metric, sk_metric, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        if dtype == torch.half and functional_metric == multilabel_ranking_average_precision:
            pytest.xfail(
                reason="multilabel_ranking_average_precision requires torch.unique which is not implemented for half"
            )
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_9 and functional_metric == multilabel_coverage_error:
            pytest.xfail(
                reason="multilabel_coverage_error requires torch.min which is only implemented for half"
                " in v1.9 or higher of torch."
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
    def test_multilabel_ranking_dtype_gpu(self, input, metric, functional_metric, sk_metric, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
            dtype=dtype,
        )


# -------------------------- Old stuff --------------------------

# def _sk_coverage_error(preds, target, sample_weight=None):
#     if sample_weight is not None:
#         sample_weight = sample_weight.numpy()
#     return sk_coverage_error(target.numpy(), preds.numpy(), sample_weight=sample_weight)


# def _sk_label_ranking(preds, target, sample_weight=None):
#     if sample_weight is not None:
#         sample_weight = sample_weight.numpy()
#     return sk_label_ranking(target.numpy(), preds.numpy(), sample_weight=sample_weight)


# def _sk_label_ranking_loss(preds, target, sample_weight=None):
#     if sample_weight is not None:
#         sample_weight = sample_weight.numpy()
#     return sk_label_ranking_loss(target.numpy(), preds.numpy(), sample_weight=sample_weight)


# @pytest.mark.parametrize(
#     "metric, functional_metric, sk_metric",
#     [
#         (CoverageError, coverage_error, _sk_coverage_error),
#         (LabelRankingAveragePrecision, label_ranking_average_precision, _sk_label_ranking),
#         (LabelRankingLoss, label_ranking_loss, _sk_label_ranking_loss),
#     ],
# )
# @pytest.mark.parametrize(
#     "preds, target",
#     [
#         (_input_mlb_logits.preds, _input_mlb_logits.target),
#         (_input_mlb_prob.preds, _input_mlb_prob.target),
#     ],
# )
# @pytest.mark.parametrize("sample_weight", [None, torch.rand(NUM_BATCHES, BATCH_SIZE)])
# class TestRanking(MetricTester):
#     @pytest.mark.parametrize("ddp", [False, True])
#     @pytest.mark.parametrize("dist_sync_on_step", [False, True])
#     def test_ranking_class(
#         self, ddp, dist_sync_on_step, preds, target, metric, functional_metric, sk_metric, sample_weight
#     ):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=metric,
#             sk_metric=sk_metric,
#             dist_sync_on_step=dist_sync_on_step,
#             fragment_kwargs=True,
#             sample_weight=sample_weight,
#         )

#     def test_ranking_functional(self, preds, target, metric, functional_metric, sk_metric, sample_weight):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=functional_metric,
#             sk_metric=sk_metric,
#             fragment_kwargs=True,
#             sample_weight=sample_weight,
#         )

#     def test_ranking_differentiability(self, preds, target, metric, functional_metric, sk_metric, sample_weight):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=metric,
#             metric_functional=functional_metric,
#         )
