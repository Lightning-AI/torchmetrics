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
from sklearn.metrics import roc_curve

from torchmetrics.classification.eer import EER, BinaryEER, MulticlassEER, MultilabelEER
from torchmetrics.functional.classification.eer import binary_eer, multiclass_eer, multilabel_eer
from torchmetrics.functional.classification.roc import binary_roc
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _reference_sklearn_eer_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    fpr, tpr, _ = roc_curve(target, preds, drop_intermediate=False)

    diff = fpr - (1 - tpr)
    idx = np.argmin(np.abs(diff))
    return (fpr[idx] + (1 - tpr[idx])) / 2


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryEER(MetricTester):
    """Test class for `BinaryEER` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_eer(self, inputs, ddp):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryEER,
            reference_metric=_reference_sklearn_eer_binary,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_eer_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_eer,
            reference_metric=partial(_reference_sklearn_eer_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_eer_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryEER,
            metric_functional=binary_eer,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_eer_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryEER,
            metric_functional=binary_eer,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_eer_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryEER,
            metric_functional=binary_eer,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_eer_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_eer(pred, true, thresholds=None)
            ap2 = binary_eer(pred, true, thresholds=threshold_fn(t.flip(0)))
            assert torch.allclose(ap1, ap2)


def _reference_sklearn_eer_multiclass(preds, target, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)

    eer = []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = roc_curve(target_temp, preds[:, i], drop_intermediate=False)
        fpr, tpr = res[0], res[1]
        fpr, tpr = [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr]]

        diff = fpr - (1 - tpr)
        idx = np.argmin(np.abs(diff))
        eer.append((fpr[idx] + (1 - tpr[idx])) / 2)
    return np.array(eer)


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassEER(MetricTester):
    """Test class for `MulticlassEER` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_eer(self, inputs, ddp):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassEER,
            reference_metric=_reference_sklearn_eer_multiclass,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    def test_multiclass_eer_functional(self, inputs):
        """Test functional implementation of metric."""
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_eer,
            reference_metric=_reference_sklearn_eer_multiclass,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    def test_multiclass_eer_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassEER,
            metric_functional=multiclass_eer,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_eer_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassEER,
            metric_functional=multiclass_eer,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_eer_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassEER,
            metric_functional=multiclass_eer,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["macro", "micro", None])
    def test_multiclass_eer_threshold_arg(self, inputs, average):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 2)) + 1e-6  # rounding will simulate binning
            ap1 = multiclass_eer(pred, true, num_classes=NUM_CLASSES, average=average, thresholds=None)
            ap2 = multiclass_eer(
                pred, true, num_classes=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2)


def _reference_sklearn_eer_multilabel(preds, target, ignore_index=None):
    eer = [_reference_sklearn_eer_binary(preds[:, i], target[:, i], ignore_index) for i in range(NUM_CLASSES)]
    return np.array(eer)


@pytest.mark.parametrize(
    "inputs", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelEER(MetricTester):
    """Test class for `MultilabelEER` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_eer(self, inputs, ddp):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelEER,
            reference_metric=_reference_sklearn_eer_multilabel,
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_eer_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_eer,
            reference_metric=partial(_reference_sklearn_eer_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_eer_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelEER,
            metric_functional=multilabel_eer,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_eer_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelEER,
            metric_functional=multilabel_eer,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_eer_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelEER,
            metric_functional=multilabel_eer,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    def test_multilabel_eer_threshold_arg(self, inputs):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            ap1 = multilabel_eer(pred, true, num_labels=NUM_CLASSES, thresholds=None)
            ap2 = multilabel_eer(pred, true, num_labels=NUM_CLASSES, thresholds=torch.linspace(0, 1, 100))
            assert torch.allclose(ap1, ap2)


@pytest.mark.parametrize(
    "metric",
    [
        BinaryEER,
        partial(MulticlassEER, num_classes=NUM_CLASSES),
        partial(MultilabelEER, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize("thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], torch.linspace(0, 1, 10)])
def test_valid_input_thresholds(recwarn, metric, thresholds):
    """Test valid formats of the threshold argument."""
    metric(thresholds=thresholds)
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryEER, {"task": "binary"}),
        (MulticlassEER, {"task": "multiclass", "num_classes": 3}),
        (MultilabelEER, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=EER):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
