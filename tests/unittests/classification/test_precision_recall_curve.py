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
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve
from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    PrecisionRecallCurve,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from torchmetrics.metric import Metric

from unittests import NUM_CLASSES
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sklearn_precision_recall_curve_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating) and not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_precision_recall_curve(target, preds)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryPrecisionRecallCurve(MetricTester):
    """Test class for `BinaryPrecisionRecallCurve` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_precision_recall_curve(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryPrecisionRecallCurve,
            reference_metric=partial(_sklearn_precision_recall_curve_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_precision_recall_curve_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_precision_recall_curve,
            reference_metric=partial(_sklearn_precision_recall_curve_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_precision_recall_curve_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionRecallCurve,
            metric_functional=binary_precision_recall_curve,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_curve_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
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
    def test_binary_precision_recall_curve_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionRecallCurve,
            metric_functional=binary_precision_recall_curve,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_precision_recall_curve_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            p1, r1, t1 = binary_precision_recall_curve(pred, true, thresholds=None)
            p2, r2, t2 = binary_precision_recall_curve(pred, true, thresholds=threshold_fn(t1))

            assert torch.allclose(p1, p2)
            assert torch.allclose(r1, r2)
            assert torch.allclose(t1, t2)

    def test_binary_error_on_wrong_dtypes(self, inputs):
        """Test that error are raised on wrong dtype."""
        preds, target = inputs

        with pytest.raises(ValueError, match="Expected argument `target` to be an int or long tensor with ground.*"):
            binary_precision_recall_curve(preds[0], target[0].to(torch.float32))

        with pytest.raises(ValueError, match="Expected argument `preds` to be an floating tensor with probability.*"):
            binary_precision_recall_curve(preds[0].long(), target[0])


def _sklearn_precision_recall_curve_multiclass(preds, target, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
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
    return [np.nan_to_num(x, nan=0.0) for x in [precision, recall, thresholds]]


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassPrecisionRecallCurve(MetricTester):
    """Test class for `MulticlassPrecisionRecallCurve` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_precision_recall_curve(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassPrecisionRecallCurve,
            reference_metric=partial(_sklearn_precision_recall_curve_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_precision_recall_curve_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_precision_recall_curve,
            reference_metric=partial(_sklearn_precision_recall_curve_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_recall_curve_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionRecallCurve,
            metric_functional=multiclass_precision_recall_curve,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_curve_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
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
    def test_multiclass_precision_recall_curve_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionRecallCurve,
            metric_functional=multiclass_precision_recall_curve,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multiclass_precision_recall_curve_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            p1, r1, t1 = multiclass_precision_recall_curve(pred, true, num_classes=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multiclass_precision_recall_curve(
                    pred, true, num_classes=NUM_CLASSES, thresholds=threshold_fn(t)
                )

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)

    def test_multiclass_error_on_wrong_dtypes(self, inputs):
        """Test that error are raised on wrong dtype."""
        preds, target = inputs

        with pytest.raises(ValueError, match="Expected argument `target` to be an int or long tensor, but got.*"):
            multiclass_precision_recall_curve(preds[0], target[0].to(torch.float32), num_classes=NUM_CLASSES)

        with pytest.raises(ValueError, match="Expected `preds` to be a float tensor, but got.*"):
            multiclass_precision_recall_curve(preds[0].long(), target[0], num_classes=NUM_CLASSES)

    @pytest.mark.parametrize("average", ["macro", "micro"])
    @pytest.mark.parametrize("thresholds", [None, 100])
    def test_multiclass_average(self, inputs, average, thresholds):
        """Test that the average argument works as expected."""
        preds, target = inputs
        output = multiclass_precision_recall_curve(
            preds[0], target[0], num_classes=NUM_CLASSES, thresholds=thresholds, average=average
        )
        assert all(isinstance(o, torch.Tensor) for o in output)
        none_output = multiclass_precision_recall_curve(
            preds[0], target[0], num_classes=NUM_CLASSES, thresholds=thresholds, average=None
        )
        if average == "macro":
            assert len(output[0]) == len(none_output[0][0]) * NUM_CLASSES
            assert len(output[1]) == len(none_output[1][0]) * NUM_CLASSES
            assert (
                len(output[2]) == (len(none_output[2][0]) if thresholds is None else len(none_output[2])) * NUM_CLASSES
            )


def _sklearn_precision_recall_curve_multilabel(preds, target, ignore_index=None):
    precision, recall, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        res = _sklearn_precision_recall_curve_binary(preds[:, i], target[:, i], ignore_index)
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])
    return precision, recall, thresholds


@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelPrecisionRecallCurve(MetricTester):
    """Test class for `MultilabelPrecisionRecallCurve` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_precision_recall_curve(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelPrecisionRecallCurve,
            reference_metric=partial(_sklearn_precision_recall_curve_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_precision_recall_curve_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_precision_recall_curve,
            reference_metric=partial(_sklearn_precision_recall_curve_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_recall_curve_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionRecallCurve,
            metric_functional=multilabel_precision_recall_curve,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_curve_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
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
    def test_multiclass_precision_recall_curve_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionRecallCurve,
            metric_functional=multilabel_precision_recall_curve,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multilabel_precision_recall_curve_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            p1, r1, t1 = multilabel_precision_recall_curve(pred, true, num_labels=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multilabel_precision_recall_curve(
                    pred, true, num_labels=NUM_CLASSES, thresholds=threshold_fn(t)
                )

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)

    def test_multilabel_error_on_wrong_dtypes(self, inputs):
        """Test that error are raised on wrong dtype."""
        preds, target = inputs

        with pytest.raises(ValueError, match="Expected argument `target` to be an int or long tensor with ground.*"):
            multilabel_precision_recall_curve(preds[0], target[0].to(torch.float32), num_labels=NUM_CLASSES)

        with pytest.raises(ValueError, match="Expected argument `preds` to be an floating tensor with probability.*"):
            multilabel_precision_recall_curve(preds[0].long(), target[0], num_labels=NUM_CLASSES)


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
    """Test valid formats of the threshold argument."""
    with pytest.warns(None) as record:
        metric(thresholds=thresholds)
    assert len(record) == 0


@pytest.mark.parametrize(
    "metric",
    [
        BinaryPrecisionRecallCurve,
        partial(MulticlassPrecisionRecallCurve, num_classes=NUM_CLASSES),
        partial(MultilabelPrecisionRecallCurve, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_empty_state_dict(metric, thresholds):
    """Test that metric have an empty state dict."""
    m = metric(thresholds=thresholds)
    assert m.state_dict() == {}, "Metric state dict should be empty."


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryPrecisionRecallCurve, {"task": "binary"}),
        (MulticlassPrecisionRecallCurve, {"task": "multiclass", "num_classes": 3}),
        (MultilabelPrecisionRecallCurve, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=PrecisionRecallCurve):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
