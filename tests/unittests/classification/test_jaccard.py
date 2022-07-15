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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import jaccard_score as sk_jaccard_index

from torchmetrics.classification.jaccard import BinaryJaccardIndex, MulticlassJaccardIndex, MultilabelJaccardIndex
from torchmetrics.functional.classification.jaccard import (
    binary_jaccard_index,
    multiclass_jaccard_index,
    multilabel_jaccard_index,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index


def _sk_jaccard_index_binary(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    return sk_jaccard_index(y_true=target, y_pred=preds)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryJaccardIndex(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_jaccard_index(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryJaccardIndex,
            sk_metric=partial(_sk_jaccard_index_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_jaccard_index_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_jaccard_index,
            sk_metric=partial(_sk_jaccard_index_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_jaccard_index_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_jaccard_index_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_jaccard_index_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_jaccard_index_multiclass(preds, target, ignore_index=None, average="macro"):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()

    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    return sk_jaccard_index(y_true=target, y_pred=preds, average=average)


@pytest.mark.parametrize("input", _multiclass_cases)
class TestMulticlassJaccardIndex(MetricTester):
    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_jaccard_index(self, input, ddp, ignore_index, average):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassJaccardIndex,
            sk_metric=partial(_sk_jaccard_index_multiclass, ignore_index=ignore_index, average=average),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_jaccard_index_functional(self, input, ignore_index, average):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_jaccard_index,
            sk_metric=partial(_sk_jaccard_index_multiclass, ignore_index=ignore_index, average=average),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
            },
        )

    def test_multiclass_jaccard_index_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_jaccard_index_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_jaccard_index_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _sk_jaccard_index_multilabel(preds, target, ignore_index=None, average="macro"):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    if ignore_index is not None:
        if average == "micro":
            return _sk_jaccard_index_binary(torch.tensor(preds), torch.tensor(target), ignore_index)
        scores, weights = [], []
        for i in range(preds.shape[1]):
            p, t = preds[:, i], target[:, i]
            if ignore_index is not None:
                idx = t == ignore_index
                t = t[~idx]
                p = p[~idx]
            confmat = sk_confusion_matrix(t, p, labels=[0, 1])
            scores.append(sk_jaccard_index(t, p))
            weights.append(confmat[1, 0] + confmat[1, 1])
        scores = np.stack(scores, axis=0)
        weights = np.stack(weights, axis=0)
        if average is None or average == "none":
            return scores
        elif average == "macro":
            return scores.mean()
        return ((scores * weights) / weights.sum()).sum()
    else:
        return sk_jaccard_index(y_true=target, y_pred=preds, average=average)


@pytest.mark.parametrize("input", _multilabel_cases)
class TestMultilabelJaccardIndex(MetricTester):
    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None])  # , -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_jaccard_index(self, input, ddp, ignore_index, average):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelJaccardIndex,
            sk_metric=partial(_sk_jaccard_index_multilabel, ignore_index=ignore_index, average=average),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_jaccard_index_functional(self, input, ignore_index, average):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_jaccard_index,
            sk_metric=partial(_sk_jaccard_index_multilabel, ignore_index=ignore_index, average=average),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
            },
        )

    def test_multilabel_jaccard_index_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_jaccard_index_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_jaccard_index_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


# -------------------------- Old stuff --------------------------

# def _sk_jaccard_binary_prob(preds, target, average=None):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_binary(preds, target, average=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multilabel_prob(preds, target, average=None):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multilabel(preds, target, average=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multiclass_prob(preds, target, average=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multiclass(preds, target, average=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multidim_multiclass_prob(preds, target, average=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# def _sk_jaccard_multidim_multiclass(preds, target, average=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


# @pytest.mark.parametrize("average", [None, "macro", "micro", "weighted"])
# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_jaccard_binary_prob, 2),
#         (_input_binary.preds, _input_binary.target, _sk_jaccard_binary, 2),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_jaccard_multilabel_prob, 2),
#         (_input_mlb.preds, _input_mlb.target, _sk_jaccard_multilabel, 2),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_jaccard_multiclass_prob, NUM_CLASSES),
#         (_input_mcls.preds, _input_mcls.target, _sk_jaccard_multiclass, NUM_CLASSES),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_jaccard_multidim_multiclass_prob, NUM_CLASSES),
#         (_input_mdmc.preds, _input_mdmc.target, _sk_jaccard_multidim_multiclass, NUM_CLASSES),
#     ],
# )
# class TestJaccardIndex(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_jaccard(self, average, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
#         # average = "macro" if reduction == "elementwise_mean" else None  # convert tags
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=JaccardIndex,
#             sk_metric=partial(sk_metric, average=average),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "average": average},
#         )

#     def test_jaccard_functional(self, average, preds, target, sk_metric, num_classes):
#         # average = "macro" if reduction == "elementwise_mean" else None  # convert tags
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=jaccard_index,
#             sk_metric=partial(sk_metric, average=average),
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "average": average},
#         )

#     def test_jaccard_differentiability(self, average, preds, target, sk_metric, num_classes):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=JaccardIndex,
#             metric_functional=jaccard_index,
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "average": average},
#         )


# @pytest.mark.parametrize(
#     ["half_ones", "average", "ignore_index", "expected"],
#     [
#         (False, "none", None, Tensor([1, 1, 1])),
#         (False, "macro", None, Tensor([1])),
#         (False, "none", 0, Tensor([1, 1])),
#         (True, "none", None, Tensor([0.5, 0.5, 0.5])),
#         (True, "macro", None, Tensor([0.5])),
#         (True, "none", 0, Tensor([2 / 3, 1 / 2])),
#     ],
# )
# def test_jaccard(half_ones, average, ignore_index, expected):
#     preds = (torch.arange(120) % 3).view(-1, 1)
#     target = (torch.arange(120) % 3).view(-1, 1)
#     if half_ones:
#         preds[:60] = 1
#     jaccard_val = jaccard_index(
#         preds=preds,
#         target=target,
#         average=average,
#         num_classes=3,
#         ignore_index=ignore_index,
#         # reduction=reduction,
#     )
#     assert torch.allclose(jaccard_val, expected, atol=1e-9)


# # test `absent_score`
# @pytest.mark.parametrize(
#     ["pred", "target", "ignore_index", "absent_score", "num_classes", "expected"],
#     [
#         # Note that -1 is used as the absent_score in almost all tests here to distinguish it from the range of valid
#         # scores the function can return ([0., 1.] range, inclusive).
#         # 2 classes, class 0 is correct everywhere, class 1 is absent.
#         ([0], [0], None, -1.0, 2, [1.0, -1.0]),
#         ([0, 0], [0, 0], None, -1.0, 2, [1.0, -1.0]),
#         # absent_score not applied if only class 0 is present and it's the only class.
#         ([0], [0], None, -1.0, 1, [1.0]),
#         # 2 classes, class 1 is correct everywhere, class 0 is absent.
#         ([1], [1], None, -1.0, 2, [-1.0, 1.0]),
#         ([1, 1], [1, 1], None, -1.0, 2, [-1.0, 1.0]),
#         # When 0 index ignored, class 0 does not get a score (not even the absent_score).
#         ([1], [1], 0, -1.0, 2, [1.0]),
#         # 3 classes. Only 0 and 2 are present, and are perfectly predicted. 1 should get absent_score.
#         ([0, 2], [0, 2], None, -1.0, 3, [1.0, -1.0, 1.0]),
#         ([2, 0], [2, 0], None, -1.0, 3, [1.0, -1.0, 1.0]),
#         # 3 classes. Only 0 and 1 are present, and are perfectly predicted. 2 should get absent_score.
#         ([0, 1], [0, 1], None, -1.0, 3, [1.0, 1.0, -1.0]),
#         ([1, 0], [1, 0], None, -1.0, 3, [1.0, 1.0, -1.0]),
#         # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in pred but not target; should not get absent_score), class
#         # 2 is absent.
#         ([0, 1], [0, 0], None, -1.0, 3, [0.5, 0.0, -1.0]),
#         # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in target but not pred; should not get absent_score), class
#         # 2 is absent.
#         ([0, 0], [0, 1], None, -1.0, 3, [0.5, 0.0, -1.0]),
#         # Sanity checks with absent_score of 1.0.
#         ([0, 2], [0, 2], None, 1.0, 3, [1.0, 1.0, 1.0]),
#         ([0, 2], [0, 2], 0, 1.0, 3, [1.0, 1.0]),
#     ],
# )
# def test_jaccard_absent_score(pred, target, ignore_index, absent_score, num_classes, expected):
#     jaccard_val = jaccard_index(
#         preds=tensor(pred),
#         target=tensor(target),
#         average=None,
#         ignore_index=ignore_index,
#         absent_score=absent_score,
#         num_classes=num_classes,
#         # reduction="none",
#     )
#     assert torch.allclose(jaccard_val, tensor(expected).to(jaccard_val))


# # example data taken from
# # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/tests/test_ranking.py
# @pytest.mark.parametrize(
#     ["pred", "target", "ignore_index", "num_classes", "average", "expected"],
#     [
#         # Ignoring an index outside of [0, num_classes-1] should have no effect.
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], None, 3, "none", [1, 1 / 2, 2 / 3]),
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], -1, 3, "none", [1, 1 / 2, 2 / 3]),
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 255, 3, "none", [1, 1 / 2, 2 / 3]),
#         # Ignoring a valid index drops only that index from the result.
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "none", [1 / 2, 2 / 3]),
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 1, 3, "none", [1, 2 / 3]),
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 2, 3, "none", [1, 1]),
#         # When reducing to mean or sum, the ignored index does not contribute to the output.
#         ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "macro", [7 / 12]),
#         # ([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "sum", [7 / 6]),
#     ],
# )
# def test_jaccard_ignore_index(pred, target, ignore_index, num_classes, average, expected):
#     jaccard_val = jaccard_index(
#         preds=tensor(pred),
#         target=tensor(target),
#         average=average,
#         ignore_index=ignore_index,
#         num_classes=num_classes,
#         # reduction=reduction,
#     )
#     assert torch.allclose(jaccard_val, tensor(expected).to(jaccard_val))
