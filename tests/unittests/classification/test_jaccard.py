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
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index


def _sk_jaccard_index_binary(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target, preds, ignore_index)
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
    target, preds = remove_ignore_index(target, preds, ignore_index)
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
    if ignore_index is None:
        return sk_jaccard_index(y_true=target, y_pred=preds, average=average)
    else:
        if average == "micro":
            return _sk_jaccard_index_binary(torch.tensor(preds), torch.tensor(target), ignore_index)
        scores, weights = [], []
        for i in range(preds.shape[1]):
            pred, true = preds[:, i], target[:, i]
            true, pred = remove_ignore_index(true, pred, ignore_index)
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            scores.append(sk_jaccard_index(true, pred))
            weights.append(confmat[1, 0] + confmat[1, 1])
        scores = np.stack(scores, axis=0)
        weights = np.stack(weights, axis=0)
        if average is None or average == "none":
            return scores
        elif average == "macro":
            return scores.mean()
        return ((scores * weights) / weights.sum()).sum()


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
