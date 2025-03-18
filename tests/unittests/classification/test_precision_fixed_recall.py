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

from torchmetrics.classification.precision_fixed_recall import (
    BinaryPrecisionAtFixedRecall,
    MulticlassPrecisionAtFixedRecall,
    MultilabelPrecisionAtFixedRecall,
    PrecisionAtFixedRecall,
)
from torchmetrics.functional.classification.precision_fixed_recall import (
    binary_precision_at_fixed_recall,
    multiclass_precision_at_fixed_recall,
    multilabel_precision_at_fixed_recall,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _precision_at_recall_x_multilabel(predictions, targets, min_recall):
    precision, recall, thresholds = sk_precision_recall_curve(targets, predictions)

    try:
        tuple_all = [(p, r, t) for p, r, t in zip(precision, recall, thresholds) if r >= min_recall]
        max_precision, _, best_threshold = max(tuple_all)
    except ValueError:
        max_precision, best_threshold = 0, 1e6

    return float(max_precision), float(best_threshold)


def _reference_sklearn_precision_at_fixed_recall_binary(preds, target, min_recall, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating) and not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return _precision_at_recall_x_multilabel(preds, target, min_recall)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryPrecisionAtFixedRecall(MetricTester):
    """Test class for `BinaryPrecisionAtFixedRecall` metric."""

    @pytest.mark.parametrize("min_recall", [0.05, 0.1, 0.3, 0.5, 0.85])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_precision_at_fixed_recall(self, inputs, ddp, min_recall, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryPrecisionAtFixedRecall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_binary, min_recall=min_recall, ignore_index=ignore_index
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_precision_at_fixed_recall_functional(self, inputs, min_recall, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_precision_at_fixed_recall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_binary, min_recall=min_recall, ignore_index=ignore_index
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_precision_at_fixed_recall_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionAtFixedRecall,
            metric_functional=binary_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_at_fixed_recall_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionAtFixedRecall,
            metric_functional=binary_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryPrecisionAtFixedRecall,
            metric_functional=binary_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    def test_binary_precision_at_fixed_recall_threshold_arg(self, inputs, min_recall):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = binary_precision_at_fixed_recall(pred, true, min_recall=min_recall, thresholds=None)
            r2, _ = binary_precision_at_fixed_recall(
                pred, true, min_recall=min_recall, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(r1, r2)


def _reference_sklearn_precision_at_fixed_recall_multiclass(preds, target, min_recall, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)

    precision, thresholds = [], []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = _precision_at_recall_x_multilabel(preds[:, i], target_temp, min_recall)
        precision.append(res[0])
        thresholds.append(res[1])
    return precision, thresholds


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassPrecisionAtFixedRecall(MetricTester):
    """Test class for `MulticlassPrecisionAtFixedRecall` metric."""

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_precision_at_fixed_recall(self, inputs, ddp, min_recall, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassPrecisionAtFixedRecall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_multiclass,
                min_recall=min_recall,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_precision_at_fixed_recall_functional(self, inputs, min_recall, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_precision_at_fixed_recall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_multiclass,
                min_recall=min_recall,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_at_fixed_recall_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionAtFixedRecall,
            metric_functional=multiclass_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_at_fixed_recall_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionAtFixedRecall,
            metric_functional=multiclass_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassPrecisionAtFixedRecall,
            metric_functional=multiclass_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    def test_multiclass_precision_at_fixed_recall_threshold_arg(self, inputs, min_recall):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = multiclass_precision_at_fixed_recall(
                pred, true, num_classes=NUM_CLASSES, min_recall=min_recall, thresholds=None
            )
            r2, _ = multiclass_precision_at_fixed_recall(
                pred, true, num_classes=NUM_CLASSES, min_recall=min_recall, thresholds=torch.linspace(0, 1, 100)
            )
            assert all(torch.allclose(r1[i], r2[i]) for i in range(len(r1)))


def _reference_sklearn_precision_at_fixed_recall_multilabel(preds, target, min_recall, ignore_index=None):
    precision, thresholds = [], []
    for i in range(NUM_CLASSES):
        res = _reference_sklearn_precision_at_fixed_recall_binary(preds[:, i], target[:, i], min_recall, ignore_index)
        precision.append(res[0])
        thresholds.append(res[1])
    return precision, thresholds


@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelPrecisionAtFixedRecall(MetricTester):
    """Test class for `MultilabelPrecisionAtFixedRecall` metric."""

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_precision_at_fixed_recall(self, inputs, ddp, min_recall, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelPrecisionAtFixedRecall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_multilabel,
                min_recall=min_recall,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_precision_at_fixed_recall_functional(self, inputs, min_recall, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_precision_at_fixed_recall,
            reference_metric=partial(
                _reference_sklearn_precision_at_fixed_recall_multilabel,
                min_recall=min_recall,
                ignore_index=ignore_index,
            ),
            metric_args={
                "min_recall": min_recall,
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_precision_at_fixed_recall_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionAtFixedRecall,
            metric_functional=multilabel_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_at_fixed_recall_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionAtFixedRecall,
            metric_functional=multilabel_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelPrecisionAtFixedRecall,
            metric_functional=multilabel_precision_at_fixed_recall,
            metric_args={"min_recall": 0.5, "thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("min_recall", [0.05, 0.5, 0.8])
    def test_multilabel_precision_at_fixed_recall_threshold_arg(self, inputs, min_recall):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            r1, _ = multilabel_precision_at_fixed_recall(
                pred, true, num_labels=NUM_CLASSES, min_recall=min_recall, thresholds=None
            )
            r2, _ = multilabel_precision_at_fixed_recall(
                pred, true, num_labels=NUM_CLASSES, min_recall=min_recall, thresholds=torch.linspace(0, 1, 100)
            )
            assert all(torch.allclose(r1[i], r2[i]) for i in range(len(r1)))


@pytest.mark.parametrize(
    "metric",
    [
        BinaryPrecisionAtFixedRecall,
        partial(MulticlassPrecisionAtFixedRecall, num_classes=NUM_CLASSES),
        partial(MultilabelPrecisionAtFixedRecall, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(recwarn, metric, thresholds):
    """Test valid formats of the threshold argument."""
    metric(min_recall=0.5, thresholds=thresholds)
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryPrecisionAtFixedRecall, {"task": "binary", "min_recall": 0.5}),
        (MulticlassPrecisionAtFixedRecall, {"task": "multiclass", "num_classes": 3, "min_recall": 0.5}),
        (MultilabelPrecisionAtFixedRecall, {"task": "multilabel", "num_labels": 3, "min_recall": 0.5}),
        (None, {"task": "not_valid_task", "min_recall": 0.5}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=PrecisionAtFixedRecall):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
