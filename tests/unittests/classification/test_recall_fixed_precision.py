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

from torchmetrics.classification.recall_fixed_precision import (
    BinaryRecallAtFixedPrecision,
    MulticlassRecallAtFixedPrecision,
    MultilabelRecallAtFixedPrecision,
    RecallAtFixedPrecision,
)
from torchmetrics.functional.classification.recall_fixed_precision import (
    binary_recall_at_fixed_precision,
    multiclass_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _recall_at_precision_x_multilabel(predictions, targets, min_precision):
    precision, recall, thresholds = sk_precision_recall_curve(targets, predictions)

    try:
        tuple_all = [(r, p, t) for p, r, t in zip(precision, recall, thresholds) if p >= min_precision]
        max_recall, _, best_threshold = max(tuple_all)
    except ValueError:
        max_recall, best_threshold = 0, 1e6

    return float(max_recall), float(best_threshold)


def _reference_sklearn_recall_at_fixed_precision_binary(preds, target, min_precision, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating) and not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return _recall_at_precision_x_multilabel(preds, target, min_precision)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryRecallAtFixedPrecision(MetricTester):
    """Test class for `BinaryRecallAtFixedPrecision` metric."""

    @pytest.mark.parametrize("min_precision", [0.05, 0.1, 0.3, 0.5, 0.85])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_recall_at_fixed_precision(self, inputs, ddp, min_precision, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryRecallAtFixedPrecision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_binary,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_recall_at_fixed_precision_functional(self, inputs, min_precision, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_recall_at_fixed_precision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_binary,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_recall_at_fixed_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryRecallAtFixedPrecision,
            metric_functional=binary_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_recall_at_fixed_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryRecallAtFixedPrecision,
            metric_functional=binary_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryRecallAtFixedPrecision,
            metric_functional=binary_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    def test_binary_recall_at_fixed_precision_threshold_arg(self, inputs, min_precision):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = binary_recall_at_fixed_precision(pred, true, min_precision=min_precision, thresholds=None)
            r2, _ = binary_recall_at_fixed_precision(
                pred, true, min_precision=min_precision, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(r1, r2)


def _reference_sklearn_recall_at_fixed_precision_multiclass(preds, target, min_precision, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)

    recall, thresholds = [], []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = _recall_at_precision_x_multilabel(preds[:, i], target_temp, min_precision)
        recall.append(res[0])
        thresholds.append(res[1])
    return recall, thresholds


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassRecallAtFixedPrecision(MetricTester):
    """Test class for `MulticlassRecallAtFixedPrecision` metric."""

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_recall_at_fixed_precision(self, inputs, ddp, min_precision, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassRecallAtFixedPrecision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_multiclass,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_recall_at_fixed_precision_functional(self, inputs, min_precision, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_recall_at_fixed_precision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_multiclass,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_recall_at_fixed_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassRecallAtFixedPrecision,
            metric_functional=multiclass_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_recall_at_fixed_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassRecallAtFixedPrecision,
            metric_functional=multiclass_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassRecallAtFixedPrecision,
            metric_functional=multiclass_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    def test_multiclass_recall_at_fixed_precision_threshold_arg(self, inputs, min_precision):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = multiclass_recall_at_fixed_precision(
                pred, true, num_classes=NUM_CLASSES, min_precision=min_precision, thresholds=None
            )
            r2, _ = multiclass_recall_at_fixed_precision(
                pred, true, num_classes=NUM_CLASSES, min_precision=min_precision, thresholds=torch.linspace(0, 1, 100)
            )
            assert all(torch.allclose(r1[i], r2[i]) for i in range(len(r1)))


def _reference_sklearn_recall_at_fixed_precision_multilabel(preds, target, min_precision, ignore_index=None):
    recall, thresholds = [], []
    for i in range(NUM_CLASSES):
        res = _reference_sklearn_recall_at_fixed_precision_binary(
            preds[:, i], target[:, i], min_precision, ignore_index
        )
        recall.append(res[0])
        thresholds.append(res[1])
    return recall, thresholds


@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelRecallAtFixedPrecision(MetricTester):
    """Test class for `MultilabelRecallAtFixedPrecision` metric."""

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_recall_at_fixed_precision(self, inputs, ddp, min_precision, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelRecallAtFixedPrecision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_multilabel,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_recall_at_fixed_precision_functional(self, inputs, min_precision, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_recall_at_fixed_precision,
            reference_metric=partial(
                _reference_sklearn_recall_at_fixed_precision_multilabel,
                min_precision=min_precision,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_precision": min_precision,
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_recall_at_fixed_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelRecallAtFixedPrecision,
            metric_functional=multilabel_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_recall_at_fixed_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelRecallAtFixedPrecision,
            metric_functional=multilabel_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelRecallAtFixedPrecision,
            metric_functional=multilabel_recall_at_fixed_precision,
            metric_args={"min_precision": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_precision", [0.05, 0.5, 0.8])
    def test_multilabel_recall_at_fixed_precision_threshold_arg(self, inputs, min_precision):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = multilabel_recall_at_fixed_precision(
                pred, true, num_labels=NUM_CLASSES, min_precision=min_precision, thresholds=None
            )
            r2, _ = multilabel_recall_at_fixed_precision(
                pred, true, num_labels=NUM_CLASSES, min_precision=min_precision, thresholds=torch.linspace(0, 1, 100)
            )
            assert all(torch.allclose(r1[i], r2[i]) for i in range(len(r1)))


@pytest.mark.parametrize(
    "metric",
    [
        BinaryRecallAtFixedPrecision,
        partial(MulticlassRecallAtFixedPrecision, num_classes=NUM_CLASSES),
        partial(MultilabelRecallAtFixedPrecision, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(recwarn, metric, thresholds):
    """Test valid formats of the threshold argument."""
    metric(min_precision=0.5, thresholds=thresholds)
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryRecallAtFixedPrecision, {"task": "binary", "min_precision": 0.5}),
        (MulticlassRecallAtFixedPrecision, {"task": "multiclass", "num_classes": 3, "min_precision": 0.5}),
        (MultilabelRecallAtFixedPrecision, {"task": "multilabel", "num_labels": 3, "min_precision": 0.5}),
        (None, {"task": "not_valid_task", "min_precision": 0.5}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=RecallAtFixedPrecision):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
