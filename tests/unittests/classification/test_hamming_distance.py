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
from sklearn.metrics import hamming_loss as sk_hamming_loss

from torchmetrics.classification.hamming import (
    BinaryHammingDistance,
    HammingDistance,
    MulticlassHammingDistance,
    MultilabelHammingDistance,
)
from torchmetrics.functional.classification.hamming import (
    binary_hamming_distance,
    multiclass_hamming_distance,
    multilabel_hamming_distance,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _reference_sklearn_hamming_loss(target, preds):
    score = sk_hamming_loss(target, preds)
    return score if not np.isnan(score) else 1.0


def _reference_sklearn_hamming_distance_binary(preds, target, ignore_index, multidim_average):
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
        target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
        return _reference_sklearn_hamming_loss(target, preds)

    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        res.append(_reference_sklearn_hamming_loss(true, pred))
    return np.stack(res)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryHammingDistance(MetricTester):
    """Test class for `BinaryHammingDistance` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_hamming_distance(self, ddp, inputs, ignore_index, multidim_average):
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
            metric_class=BinaryHammingDistance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_binary, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_hamming_distance_functional(self, inputs, ignore_index, multidim_average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_hamming_distance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_binary, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_hamming_distance_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hamming_distance_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hamming_distance_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_hamming_distance_multiclass_global(preds, target, ignore_index, average):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    if average == "micro":
        return _reference_sklearn_hamming_loss(target, preds)
    confmat = sk_confusion_matrix(y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES)))
    hamming_per_class = 1 - confmat.diagonal() / confmat.sum(axis=1)
    hamming_per_class[np.isnan(hamming_per_class)] = 1.0
    if average == "macro":
        hamming_per_class = hamming_per_class[
            (np.bincount(preds, minlength=NUM_CLASSES) + np.bincount(target, minlength=NUM_CLASSES)) != 0.0
        ]
        return hamming_per_class.mean()
    if average == "weighted":
        weights = confmat.sum(1)
        return ((weights * hamming_per_class) / weights.sum()).sum()
    return hamming_per_class


def _reference_sklearn_hamming_distance_multiclass_local(preds, target, ignore_index, average):
    preds = preds.numpy()
    target = target.numpy()
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        if average == "micro":
            res.append(_reference_sklearn_hamming_loss(true, pred))
        else:
            confmat = sk_confusion_matrix(true, pred, labels=list(range(NUM_CLASSES)))
            hamming_per_class = 1 - confmat.diagonal() / confmat.sum(axis=1)
            hamming_per_class[np.isnan(hamming_per_class)] = 1.0
            if average == "macro":
                hamming_per_class = hamming_per_class[
                    (np.bincount(pred, minlength=NUM_CLASSES) + np.bincount(true, minlength=NUM_CLASSES)) != 0.0
                ]
                res.append(hamming_per_class.mean() if len(hamming_per_class) > 0 else 0.0)
            elif average == "weighted":
                weights = confmat.sum(1)
                score = ((weights * hamming_per_class) / weights.sum()).sum()
                res.append(0.0 if np.isnan(score) else score)
            else:
                res.append(hamming_per_class)
    return np.stack(res, 0)


def _reference_sklearn_hamming_distance_multiclass(preds, target, ignore_index, multidim_average, average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    if multidim_average == "global":
        return _reference_sklearn_hamming_distance_multiclass_global(preds, target, ignore_index, average)
    return _reference_sklearn_hamming_distance_multiclass_local(preds, target, ignore_index, average)


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassHammingDistance(MetricTester):
    """Test class for `MulticlassHammingDistance` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_hamming_distance(self, ddp, inputs, ignore_index, multidim_average, average):
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
            metric_class=MulticlassHammingDistance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_multiclass,
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
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multiclass_hamming_distance_functional(self, inputs, ignore_index, multidim_average, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_hamming_distance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_multiclass,
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

    def test_multiclass_hamming_distance_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassHammingDistance,
            metric_functional=multiclass_hamming_distance,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_hamming_distance_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassHammingDistance,
            metric_functional=multiclass_hamming_distance,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_hamming_distance_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassHammingDistance,
            metric_functional=multiclass_hamming_distance,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _reference_sklearn_hamming_distance_multilabel_global(preds, target, ignore_index, average):
    if average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
        return _reference_sklearn_hamming_loss(target, preds)

    hamming, weights = [], []
    for i in range(preds.shape[1]):
        pred, true = preds[:, i].flatten(), target[:, i].flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
        hamming.append(_reference_sklearn_hamming_loss(true, pred))
        weights.append(confmat[1, 1] + confmat[1, 0])
    res = np.stack(hamming, axis=0)

    if average == "macro":
        return res.mean(0)
    if average == "weighted":
        weights = np.stack(weights, 0).astype(float)
        weights_norm = weights.sum(-1, keepdims=True)
        weights_norm[weights_norm == 0] = 1.0
        return ((weights * res) / weights_norm).sum(-1)
    if average is None or average == "none":
        return res
    return None


def _reference_sklearn_hamming_distance_multilabel_local(preds, target, ignore_index, average):
    hamming, weights = [], []
    for i in range(preds.shape[0]):
        if average == "micro":
            pred, true = preds[i].flatten(), target[i].flatten()
            true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
            hamming.append(_reference_sklearn_hamming_loss(true, pred))
        else:
            scores, w = [], []
            for j in range(preds.shape[1]):
                pred, true = preds[i, j], target[i, j]
                true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
                scores.append(_reference_sklearn_hamming_loss(true, pred))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                w.append(confmat[1, 1] + confmat[1, 0])
            hamming.append(np.stack(scores))
            weights.append(np.stack(w))
    if average == "micro":
        return np.array(hamming)
    res = np.stack(hamming, 0)
    if average == "macro":
        return res.mean(-1)
    if average == "weighted":
        weights = np.stack(weights, 0).astype(float)
        weights_norm = weights.sum(-1, keepdims=True)
        weights_norm[weights_norm == 0] = 1.0
        return ((weights * res) / weights_norm).sum(-1)
    if average is None or average == "none":
        return res
    return None


def _reference_sklearn_hamming_distance_multilabel(preds, target, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)

    if multidim_average == "global":
        return _reference_sklearn_hamming_distance_multilabel_global(preds, target, ignore_index, average)
    return _reference_sklearn_hamming_distance_multilabel_local(preds, target, ignore_index, average)


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelHammingDistance(MetricTester):
    """Test class for `MultilabelHammingDistance` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multilabel_hamming_distance(self, ddp, inputs, ignore_index, multidim_average, average):
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
            metric_class=MultilabelHammingDistance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_multilabel,
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
    def test_multilabel_hamming_distance_functional(self, inputs, ignore_index, multidim_average, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_hamming_distance,
            reference_metric=partial(
                _reference_sklearn_hamming_distance_multilabel,
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

    def test_multilabel_hamming_distance_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelHammingDistance,
            metric_functional=multilabel_hamming_distance,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_hamming_distance_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelHammingDistance,
            metric_functional=multilabel_hamming_distance,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_hamming_distance_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelHammingDistance,
            metric_functional=multilabel_hamming_distance,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryHammingDistance, {"task": "binary"}),
        (MulticlassHammingDistance, {"task": "multiclass", "num_classes": 3}),
        (MultilabelHammingDistance, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=HammingDistance):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
