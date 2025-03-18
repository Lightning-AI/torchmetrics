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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch import Tensor, tensor

from torchmetrics.classification.negative_predictive_value import (
    BinaryNegativePredictiveValue,
    MulticlassNegativePredictiveValue,
    MultilabelNegativePredictiveValue,
    NegativePredictiveValue,
)
from torchmetrics.functional.classification.negative_predictive_value import (
    binary_negative_predictive_value,
    multiclass_negative_predictive_value,
    multilabel_negative_predictive_value,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _calc_negative_predictive_value(tn, fn):
    """Safely calculate negative_predictive_value."""
    denom = tn + fn
    if np.isscalar(tn):
        denom = 1.0 if denom == 0 else denom
    else:
        denom[denom == 0] = 1.0
    return tn / denom


def _reference_negative_predictive_value_binary(preds, target, ignore_index, multidim_average):
    if multidim_average == "global":
        preds = preds.view(-1).numpy()
        target = target.view(-1).numpy()
    else:
        preds = preds.numpy()
        target = target.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    if multidim_average == "global":
        if ignore_index is not None:
            idx = target == ignore_index
            target = target[~idx]
            preds = preds[~idx]
        tn, _, fn, _ = sk_confusion_matrix(y_true=target, y_pred=preds, labels=[0, 1]).ravel()
        return _calc_negative_predictive_value(tn, fn)

    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        if ignore_index is not None:
            idx = true == ignore_index
            true = true[~idx]
            pred = pred[~idx]
        tn, _, fn, _ = sk_confusion_matrix(y_true=true, y_pred=pred, labels=[0, 1]).ravel()
        res.append(_calc_negative_predictive_value(tn, fn))
    return np.stack(res)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryNegativePredictiveValue(MetricTester):
    """Test class for `BinaryNegativePredictiveValue` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_negative_predictive_value(self, ddp, inputs, ignore_index, multidim_average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryNegativePredictiveValue,
            reference_metric=partial(
                _reference_negative_predictive_value_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_negative_predictive_value_functional(self, inputs, ignore_index, multidim_average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_negative_predictive_value,
            reference_metric=partial(
                _reference_negative_predictive_value_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_negative_predictive_value_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryNegativePredictiveValue,
            metric_functional=binary_negative_predictive_value,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_negative_predictive_value_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryNegativePredictiveValue,
            metric_functional=binary_negative_predictive_value,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_negative_predictive_value_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryNegativePredictiveValue,
            metric_functional=binary_negative_predictive_value,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_negative_predictive_value_multiclass_global(preds, target, ignore_index, average):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()

    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    confmat = sk_confusion_matrix(y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES)))
    tp = np.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    tn = confmat.sum() - (fp + fn + tp)

    if average == "micro":
        return _calc_negative_predictive_value(tn.sum(), fn.sum())

    res = _calc_negative_predictive_value(tn, fn)
    if average == "macro":
        res = res[(np.bincount(preds, minlength=NUM_CLASSES) + np.bincount(target, minlength=NUM_CLASSES)) != 0.0]
        return res.mean(0)
    if average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
    if average is None or average == "none":
        return res
    return None


def _reference_negative_predictive_value_multiclass_local(preds, target, ignore_index, average):
    preds = preds.numpy()
    target = target.numpy()

    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()

        if ignore_index is not None:
            idx = true == ignore_index
            true = true[~idx]
            pred = pred[~idx]
        confmat = sk_confusion_matrix(y_true=true, y_pred=pred, labels=list(range(NUM_CLASSES)))
        tp = np.diag(confmat)
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
        if average == "micro":
            res.append(_calc_negative_predictive_value(tn.sum(), fn.sum()))

        r = _calc_negative_predictive_value(tn, fn)
        if average == "macro":
            r = r[(np.bincount(pred, minlength=NUM_CLASSES) + np.bincount(true, minlength=NUM_CLASSES)) != 0.0]
            res.append(r.mean(0) if len(r) > 0 else 0.0)
        elif average == "weighted":
            w = tp + fn
            res.append((r * (w / w.sum()).reshape(-1, 1)).sum(0))
        elif average is None or average == "none":
            res.append(r)
    return np.stack(res, 0)


def _reference_negative_predictive_value_multiclass(preds, target, ignore_index, multidim_average, average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    if multidim_average == "global":
        return _reference_negative_predictive_value_multiclass_global(preds, target, ignore_index, average)
    return _reference_negative_predictive_value_multiclass_local(preds, target, ignore_index, average)


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassNegativePredictiveValue(MetricTester):
    """Test class for `MulticlassNegativePredictiveValue` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_negative_predictive_value(self, ddp, inputs, ignore_index, multidim_average, average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassNegativePredictiveValue,
            reference_metric=partial(
                _reference_negative_predictive_value_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multiclass_negative_predictive_value_functional(self, inputs, ignore_index, multidim_average, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_negative_predictive_value,
            reference_metric=partial(
                _reference_negative_predictive_value_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    def test_multiclass_negative_predictive_value_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassNegativePredictiveValue,
            metric_functional=multiclass_negative_predictive_value,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_negative_predictive_value_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassNegativePredictiveValue,
            metric_functional=multiclass_negative_predictive_value,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_negative_predictive_value_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassNegativePredictiveValue,
            metric_functional=multiclass_negative_predictive_value,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


_mc_k_target = tensor([0, 1, 2])
_mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])


@pytest.mark.parametrize(
    ("k", "preds", "target", "average", "expected_spec"),
    [
        (1, _mc_k_preds, _mc_k_target, "micro", tensor(5 / 6)),
        (2, _mc_k_preds, _mc_k_target, "micro", tensor(1)),
    ],
)
def test_top_k(k: int, preds: Tensor, target: Tensor, average: str, expected_spec: Tensor):
    """A simple test to check that top_k works as expected."""
    class_metric = MulticlassNegativePredictiveValue(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    assert torch.equal(class_metric.compute(), expected_spec)
    assert torch.equal(
        multiclass_negative_predictive_value(preds, target, top_k=k, average=average, num_classes=3), expected_spec
    )


def _reference_negative_predictive_value_multilabel_global(preds, target, ignore_index, average):
    tns, fns = [], []
    for i in range(preds.shape[1]):
        p, t = preds[:, i].flatten(), target[:, i].flatten()
        if ignore_index is not None:
            idx = t == ignore_index
            t = t[~idx]
            p = p[~idx]
        tn, _, fn, _ = sk_confusion_matrix(t, p, labels=[0, 1]).ravel()
        tns.append(tn)
        fns.append(fn)

    tn = np.array(tns)
    fn = np.array(fns)
    if average == "micro":
        return _calc_negative_predictive_value(tn.sum(), fn.sum())

    res = _calc_negative_predictive_value(tn, fn)
    if average == "macro":
        return res.mean(0)
    if average == "weighted":
        w = res[:, 0] + res[:, 3]
        return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
    if average is None or average == "none":
        return res
    return None


def _reference_negative_predictive_value_multilabel_local(preds, target, ignore_index, average):
    negative_predictive_value = []
    for i in range(preds.shape[0]):
        tns, fns = [], []
        for j in range(preds.shape[1]):
            pred, true = preds[i, j], target[i, j]
            if ignore_index is not None:
                idx = true == ignore_index
                true = true[~idx]
                pred = pred[~idx]
            tn, _, fn, _ = sk_confusion_matrix(true, pred, labels=[0, 1]).ravel()
            tns.append(tn)
            fns.append(fn)
        tn = np.array(tns)
        fn = np.array(fns)
        if average == "micro":
            negative_predictive_value.append(_calc_negative_predictive_value(tn.sum(), fn.sum()))
        else:
            negative_predictive_value.append(_calc_negative_predictive_value(tn, fn))

    res = np.stack(negative_predictive_value, 0)
    if average == "micro" or average is None or average == "none":
        return res
    if average == "macro":
        return res.mean(-1)
    if average == "weighted":
        w = res[:, 0, :] + res[:, 3, :]
        return (res * (w / w.sum())[:, np.newaxis]).sum(-1)
    if average is None or average == "none":
        return np.moveaxis(res, 1, -1)
    return None


def _reference_negative_predictive_value_multilabel(preds, target, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if multidim_average == "global":
        return _reference_negative_predictive_value_multilabel_global(preds, target, ignore_index, average)
    return _reference_negative_predictive_value_multilabel_local(preds, target, ignore_index, average)


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelNegativePredictiveValue(MetricTester):
    """Test class for `MultilabelNegativePredictiveValue` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multilabel_negative_predictive_value(self, ddp, inputs, ignore_index, multidim_average, average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelNegativePredictiveValue,
            reference_metric=partial(
                _reference_negative_predictive_value_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multilabel_negative_predictive_value_functional(self, inputs, ignore_index, multidim_average, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_negative_predictive_value,
            reference_metric=partial(
                _reference_negative_predictive_value_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    def test_multilabel_negative_predictive_value_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelNegativePredictiveValue,
            metric_functional=multilabel_negative_predictive_value,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_negative_predictive_value_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelNegativePredictiveValue,
            metric_functional=multilabel_negative_predictive_value,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_negative_predictive_value_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelNegativePredictiveValue,
            metric_functional=multilabel_negative_predictive_value,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_corner_cases():
    """Test corner cases for negative predictive value metric."""
    # simulate the output of a perfect predictor (i.e. preds == target)
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = target

    metric = MulticlassNegativePredictiveValue(num_classes=3, average="none", ignore_index=0)
    res = metric(preds, target)
    assert torch.allclose(res, torch.tensor([1.0, 1.0, 1.0]))

    metric = MulticlassNegativePredictiveValue(num_classes=3, average="macro", ignore_index=0)
    res = metric(preds, target)
    assert res == 1.0


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryNegativePredictiveValue, {"task": "binary"}),
        (MulticlassNegativePredictiveValue, {"task": "multiclass", "num_classes": 3}),
        (MultilabelNegativePredictiveValue, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=NegativePredictiveValue):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
