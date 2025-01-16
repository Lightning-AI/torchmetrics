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
from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef

from torchmetrics.classification.matthews_corrcoef import (
    BinaryMatthewsCorrCoef,
    MatthewsCorrCoef,
    MulticlassMatthewsCorrCoef,
    MultilabelMatthewsCorrCoef,
)
from torchmetrics.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _reference_sklearn_matthews_corrcoef_binary(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryMatthewsCorrCoef(MetricTester):
    """Test class for `BinaryMatthewsCorrCoef` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_matthews_corrcoef(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryMatthewsCorrCoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_matthews_corrcoef_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_matthews_corrcoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_matthews_corrcoef_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_matthews_corrcoef_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_matthews_corrcoef_multiclass(preds, target, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassMatthewsCorrCoef(MetricTester):
    """Test class for `MulticlassMatthewsCorrCoef` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_matthews_corrcoef(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassMatthewsCorrCoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_multiclass, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_matthews_corrcoef_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_matthews_corrcoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_multiclass, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_matthews_corrcoef_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_matthews_corrcoef_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _reference_sklearn_matthews_corrcoef_multilabel(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelMatthewsCorrCoef(MetricTester):
    """Test class for `MultilabelMatthewsCorrCoef` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_matthews_corrcoef(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelMatthewsCorrCoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_multilabel, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_matthews_corrcoef_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_matthews_corrcoef,
            reference_metric=partial(_reference_sklearn_matthews_corrcoef_multilabel, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_matthews_corrcoef_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_matthews_corrcoef_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_zero_case_in_multiclass():
    """Cases where the denominator in the matthews corrcoef is 0, the score should return 0."""
    # Example where neither 1 or 2 is present in the target tensor
    out = multiclass_matthews_corrcoef(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]), 3)
    assert out == 0.0


@pytest.mark.parametrize(
    ("metric_fn", "preds", "target", "expected"),
    [
        (binary_matthews_corrcoef, torch.zeros(10), torch.zeros(10), 1.0),
        (binary_matthews_corrcoef, torch.ones(10), torch.ones(10), 1.0),
        (
            binary_matthews_corrcoef,
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            0.0,
        ),
        (
            binary_matthews_corrcoef,
            torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            0.0,
        ),
        (binary_matthews_corrcoef, torch.zeros(10), torch.ones(10), -1.0),
        (binary_matthews_corrcoef, torch.ones(10), torch.zeros(10), -1.0),
        (
            partial(multilabel_matthews_corrcoef, num_labels=NUM_CLASSES),
            torch.zeros(10, NUM_CLASSES).long(),
            torch.zeros(10, NUM_CLASSES).long(),
            1.0,
        ),
        (
            partial(multilabel_matthews_corrcoef, num_labels=NUM_CLASSES),
            torch.ones(10, NUM_CLASSES).long(),
            torch.ones(10, NUM_CLASSES).long(),
            1.0,
        ),
        (
            partial(multilabel_matthews_corrcoef, num_labels=NUM_CLASSES),
            torch.zeros(10, NUM_CLASSES).long(),
            torch.ones(10, NUM_CLASSES).long(),
            -1.0,
        ),
        (
            partial(multilabel_matthews_corrcoef, num_labels=NUM_CLASSES),
            torch.ones(10, NUM_CLASSES).long(),
            torch.zeros(10, NUM_CLASSES).long(),
            -1.0,
        ),
    ],
)
def test_corner_cases(metric_fn, preds, target, expected):
    """Test the corner cases of perfect classifiers or completely random classifiers that they work as expected."""
    out = metric_fn(preds, target)
    assert out == expected


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryMatthewsCorrCoef, {"task": "binary"}),
        (MulticlassMatthewsCorrCoef, {"task": "multiclass", "num_classes": 3}),
        (MultilabelMatthewsCorrCoef, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=MatthewsCorrCoef):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
