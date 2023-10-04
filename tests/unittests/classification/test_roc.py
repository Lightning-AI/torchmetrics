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
from scipy.special import softmax
from sklearn.metrics import roc_curve as sk_roc_curve
from torchmetrics.classification.roc import ROC, BinaryROC, MulticlassROC, MultilabelROC
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc
from torchmetrics.metric import Metric

from unittests import NUM_CLASSES
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sklearn_roc_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating) and not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    fpr, tpr, thresholds = sk_roc_curve(target, preds, drop_intermediate=False)
    thresholds[0] = 1.0
    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryROC(MetricTester):
    """Test class for `BinaryROC` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_roc(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryROC,
            reference_metric=partial(_sklearn_roc_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_roc_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_roc,
            reference_metric=partial(_sklearn_roc_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_roc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_roc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_roc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_roc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            p1, r1, t1 = binary_roc(pred, true, thresholds=None)
            p2, r2, t2 = binary_roc(pred, true, thresholds=threshold_fn(t1.flip(0)))
            assert torch.allclose(p1, p2)
            assert torch.allclose(r1, r2)
            assert torch.allclose(t1, t2)


def _sklearn_roc_multiclass(preds, target, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target, preds, ignore_index)

    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = sk_roc_curve(target_temp, preds[:, i], drop_intermediate=False)
        res[2][0] = 1.0

        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassROC(MetricTester):
    """Test class for `MulticlassROC` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_roc(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassROC,
            reference_metric=partial(_sklearn_roc_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_roc_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_roc,
            reference_metric=partial(_sklearn_roc_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_roc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multiclass_roc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            p1, r1, t1 = multiclass_roc(pred, true, num_classes=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multiclass_roc(pred, true, num_classes=NUM_CLASSES, thresholds=threshold_fn(t.flip(0)))

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)

    @pytest.mark.parametrize("average", ["macro", "micro"])
    @pytest.mark.parametrize("thresholds", [None, 100])
    def test_multiclass_average(self, inputs, average, thresholds):
        """Test that the average argument works as expected."""
        preds, target = inputs
        output = multiclass_roc(preds[0], target[0], num_classes=NUM_CLASSES, thresholds=thresholds, average=average)
        assert all(isinstance(o, torch.Tensor) for o in output)
        none_output = multiclass_roc(preds[0], target[0], num_classes=NUM_CLASSES, thresholds=thresholds, average=None)
        if average == "macro":
            assert len(output[0]) == len(none_output[0][0]) * NUM_CLASSES
            assert len(output[1]) == len(none_output[1][0]) * NUM_CLASSES
            assert (
                len(output[2]) == (len(none_output[2][0]) if thresholds is None else len(none_output[2])) * NUM_CLASSES
            )


def _sklearn_roc_multilabel(preds, target, ignore_index=None):
    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        res = _sklearn_roc_binary(preds[:, i], target[:, i], ignore_index)
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return fpr, tpr, thresholds


@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelROC(MetricTester):
    """Test class for `MultilabelROC` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_roc(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelROC,
            reference_metric=partial(_sklearn_roc_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_roc_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_roc,
            reference_metric=partial(_sklearn_roc_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_roc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_roc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multilabel_roc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            p1, r1, t1 = multilabel_roc(pred, true, num_labels=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multilabel_roc(pred, true, num_labels=NUM_CLASSES, thresholds=threshold_fn(t.flip(0)))

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)


@pytest.mark.parametrize(
    "metric",
    [
        BinaryROC,
        partial(MulticlassROC, num_classes=NUM_CLASSES),
        partial(MultilabelROC, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(metric, thresholds):
    """Test valid formats of the threshold argument."""
    with pytest.warns(None) as record:
        metric(thresholds=thresholds)
    assert len(record) == 0


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryROC, {"task": "binary"}),
        (MulticlassROC, {"task": "multiclass", "num_classes": 3}),
        (MultilabelROC, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=ROC):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
