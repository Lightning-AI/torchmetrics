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
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_precision_recall_curve_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_precision_recall_curve(target, preds)


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryPrecisionRecallCurve(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_precision_recall_curve(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryPrecisionRecallCurve,
            sk_metric=partial(_sk_precision_recall_curve_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_precision_recall_curve_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_precision_recall_curve,
            sk_metric=partial(_sk_precision_recall_curve_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_precision_recall_curve_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionRecallCurve,
            metric_functional=binary_precision_recall_curve,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_curve_dtype_cpu(self, input, dtype):
        preds, target = input
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionRecallCurve,
            metric_functional=binary_precision_recall_curve,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_curve_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionRecallCurve,
            metric_functional=binary_precision_recall_curve,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_precision_recall_curve_threshold_arg(self, input, threshold_fn):
        preds, target = input

        for pred, true in zip(preds, target):
            p1, r1, t1 = binary_precision_recall_curve(pred, true, thresholds=None)
            p2, r2, t2 = binary_precision_recall_curve(pred, true, thresholds=threshold_fn(t1))

            assert torch.allclose(p1, p2)
            assert torch.allclose(r1, r2)
            assert torch.allclose(t1, t2)


def _sk_precision_recall_curve_multiclass(preds, target, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target, preds, ignore_index)

    precision, recall, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = sk_precision_recall_curve(target_temp, preds[:, i])
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])
    # return precision, recall, thresholds
    return [np.nan_to_num(x, nan=0.0) for x in [precision, recall, thresholds]]


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassPrecisionRecallCurve(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_precision_recall_curve(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassPrecisionRecallCurve,
            sk_metric=partial(_sk_precision_recall_curve_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_precision_recall_curve_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_precision_recall_curve,
            sk_metric=partial(_sk_precision_recall_curve_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_recall_curve_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionRecallCurve,
            metric_functional=multiclass_precision_recall_curve,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_curve_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionRecallCurve,
            metric_functional=multiclass_precision_recall_curve,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_curve_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionRecallCurve,
            metric_functional=multiclass_precision_recall_curve,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multiclass_precision_recall_curve_threshold_arg(self, input, threshold_fn):
        preds, target = input
        for pred, true in zip(preds, target):
            p1, r1, t1 = multiclass_precision_recall_curve(pred, true, num_classes=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multiclass_precision_recall_curve(
                    pred, true, num_classes=NUM_CLASSES, thresholds=threshold_fn(t)
                )

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)


def _sk_precision_recall_curve_multilabel(preds, target, ignore_index=None):
    precision, recall, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        res = _sk_precision_recall_curve_binary(preds[:, i], target[:, i], ignore_index)
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])
    return precision, recall, thresholds


@pytest.mark.parametrize(
    "input", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelPrecisionRecallCurve(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_precision_recall_curve(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelPrecisionRecallCurve,
            sk_metric=partial(_sk_precision_recall_curve_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_precision_recall_curve_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_precision_recall_curve,
            sk_metric=partial(_sk_precision_recall_curve_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_recall_curve_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionRecallCurve,
            metric_functional=multilabel_precision_recall_curve,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_curve_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionRecallCurve,
            metric_functional=multilabel_precision_recall_curve,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_curve_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionRecallCurve,
            metric_functional=multilabel_precision_recall_curve,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multilabel_precision_recall_curve_threshold_arg(self, input, threshold_fn):
        preds, target = input
        for pred, true in zip(preds, target):
            p1, r1, t1 = multilabel_precision_recall_curve(pred, true, num_labels=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multilabel_precision_recall_curve(
                    pred, true, num_labels=NUM_CLASSES, thresholds=threshold_fn(t)
                )

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)


@pytest.mark.parametrize(
    "metric",
    [
        BinaryPrecisionRecallCurve,
        partial(MulticlassPrecisionRecallCurve, num_classes=NUM_CLASSES),
        partial(MultilabelPrecisionRecallCurve, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(metric, thresholds):
    """test valid formats of the threshold argument."""
    with pytest.warns(None) as record:
        metric(thresholds=thresholds)
    assert len(record) == 0
