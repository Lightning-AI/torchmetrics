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
from scipy.special import softmax
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from torchmetrics.classification.auroc import BinaryAUROC, MulticlassAUROC, MultilabelAUROC
from torchmetrics.functional.classification.auroc import binary_auroc, multiclass_auroc, multilabel_auroc
from torchmetrics.functional.classification.roc import binary_roc
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_auroc_binary(preds, target, max_fpr=None, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_roc_auc_score(target, preds, max_fpr=max_fpr)


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryAUROC(MetricTester):
    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_auroc(self, input, ddp, max_fpr, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAUROC,
            sk_metric=partial(_sk_auroc_binary, max_fpr=max_fpr, ignore_index=ignore_index),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_auroc_functional(self, input, max_fpr, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_auroc,
            sk_metric=partial(_sk_auroc_binary, max_fpr=max_fpr, ignore_index=ignore_index),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_auroc_dtype_cpu(self, input, dtype):
        preds, target = input

        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_auroc_threshold_arg(self, input, threshold_fn):
        preds, target = input

        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_auroc(pred, true, thresholds=None)
            ap2 = binary_auroc(pred, true, thresholds=threshold_fn(t.flip(0)))
            assert torch.allclose(ap1, ap2)


def _sk_auroc_multiclass(preds, target, average="macro", ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_roc_auc_score(target, preds, average=average, multi_class="ovr", labels=list(range(NUM_CLASSES)))


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassAUROC(MetricTester):
    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_auroc(self, input, average, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAUROC,
            sk_metric=partial(_sk_auroc_multiclass, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_auroc_functional(self, input, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_auroc,
            sk_metric=partial(_sk_auroc_multiclass, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_cpu(self, input, dtype):
        preds, target = input

        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    def test_multiclass_auroc_threshold_arg(self, input, average):
        preds, target = input
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 2)) + 1e-6  # rounding will simulate binning
            ap1 = multiclass_auroc(pred, true, num_classes=NUM_CLASSES, average=average, thresholds=None)
            ap2 = multiclass_auroc(
                pred, true, num_classes=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2)


def _sk_auroc_multilabel(preds, target, average="macro", ignore_index=None):
    if ignore_index is None:
        if preds.ndim > 2:
            target = target.transpose(2, 1).reshape(-1, NUM_CLASSES)
            preds = preds.transpose(2, 1).reshape(-1, NUM_CLASSES)
        target = target.numpy()
        preds = preds.numpy()
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        return sk_roc_auc_score(target, preds, average=average, max_fpr=None)
    if average == "micro":
        return _sk_auroc_binary(preds.flatten(), target.flatten(), max_fpr=None, ignore_index=ignore_index)
    res = []
    for i in range(NUM_CLASSES):
        res.append(_sk_auroc_binary(preds[:, i], target[:, i], max_fpr=None, ignore_index=ignore_index))
    if average == "macro":
        return np.array(res)[~np.isnan(res)].mean()
    if average == "weighted":
        weights = ((target == 1).sum([0, 2]) if target.ndim == 3 else (target == 1).sum(0)).numpy()
        weights = weights / sum(weights)
        return (np.array(res) * weights)[~np.isnan(res)].sum()
    return res


@pytest.mark.parametrize(
    "input", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelAUROC(MetricTester):
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_auroc(self, input, ddp, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelAUROC,
            sk_metric=partial(_sk_auroc_multilabel, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_auroc_functional(self, input, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_auroc,
            sk_metric=partial(_sk_auroc_multilabel, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_auroc_dtype_cpu(self, input, dtype):
        preds, target = input

        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_auroc_threshold_arg(self, input, average):
        preds, target = input
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            ap1 = multilabel_auroc(pred, true, num_labels=NUM_CLASSES, average=average, thresholds=None)
            ap2 = multilabel_auroc(
                pred, true, num_labels=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2)


@pytest.mark.parametrize(
    "metric",
    [
        BinaryAUROC,
        partial(MulticlassAUROC, num_classes=NUM_CLASSES),
        partial(MultilabelAUROC, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(metric, thresholds):
    """test valid formats of the threshold argument."""
    with pytest.warns(None) as record:
        metric(thresholds=thresholds)
    assert len(record) == 0
