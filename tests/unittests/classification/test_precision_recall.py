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
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score
from torch import Tensor, tensor

from torchmetrics.classification.precision_recall import (
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
    Precision,
    Recall,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _reference_sklearn_precision_recall_binary(
    preds, target, sk_fn, ignore_index, multidim_average, zero_division=0, prob_threshold: float = THRESHOLD
):
    if multidim_average == "global":
        preds = preds.view(-1).numpy()
        target = target.view(-1).numpy()
    else:
        preds = preds.numpy()
        target = target.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= prob_threshold).astype(np.uint8)

    if multidim_average == "global":
        target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
        return sk_fn(target, preds, zero_division=zero_division)

    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        res.append(sk_fn(true, pred, zero_division=zero_division))
    return np.stack(res)


@pytest.mark.parametrize("inputs", _binary_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (BinaryPrecision, binary_precision, sk_precision_score),
        (BinaryRecall, binary_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestBinaryPrecisionRecall(MetricTester):
    """Test class for `BinaryPrecisionRecall` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_precision_recall(
        self, ddp, inputs, module, functional, compare, ignore_index, multidim_average, zero_division
    ):
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_precision_recall_binary,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_precision_recall_functional(
        self, inputs, module, functional, compare, ignore_index, multidim_average, zero_division
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_precision_recall_binary,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "zero_division": zero_division,
            },
        )

    def test_binary_precision_recall_differentiability(self, inputs, module, functional, compare):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_half_cpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_precision_recall_multiclass(
    preds, target, sk_fn, ignore_index, multidim_average, average, zero_division=0, num_classes: int = NUM_CLASSES
):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)

    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()
        target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
        return sk_fn(
            target,
            preds,
            average=average,
            labels=list(range(num_classes)) if average is None else None,
            zero_division=zero_division,
        )

    preds = preds.numpy()
    target = target.numpy()
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        if len(pred) == 0 and average == "weighted":
            # The result of sk_fn([], [], labels=None, average="weighted", zero_division=zero_division)
            # varies depending on the sklearn version:
            # 1.2 -> the value of zero_division
            # 1.3 -> nan
            # 1.4 -> nan
            # To avoid breaking some test cases by this behavior,
            # hard coded to return 0 in this special case.
            r = 0.0
        else:
            r = sk_fn(
                true,
                pred,
                average=average,
                labels=list(range(num_classes)) if average is None else None,
                zero_division=zero_division,
            )
        res.append(0.0 if np.isnan(r).any() else r)

    return np.stack(res, 0)


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (MulticlassPrecision, multiclass_precision, sk_precision_score),
        (MulticlassRecall, multiclass_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestMulticlassPrecisionRecall(MetricTester):
    """Test class for `MulticlassPrecisionRecall` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_precision_recall(
        self,
        ddp,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        average,
        zero_division,
    ):
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_precision_recall_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_precision_recall_functional(
        self, inputs, module, functional, compare, ignore_index, multidim_average, average, zero_division
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_precision_recall_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
                "zero_division": zero_division,
            },
        )

    def test_multiclass_precision_recall_differentiability(self, inputs, module, functional, compare):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_half_cpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


_mc_k_target = tensor([0, 1, 2])
_mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])

_mc_k_targets2 = tensor([0, 0, 2])
_mc_k_preds2 = tensor([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]])

_mc_k_target3 = tensor([0, 1, 2, 0])
_mc_k_preds3 = tensor([
    [0.1, 0.2, 0.7],
    [0.4, 0.4, 0.2],
    [0.3, 0.3, 0.4],
    [0.3, 0.3, 0.4],
])


@pytest.mark.parametrize(
    ("metric_class", "metric_fn"), [(MulticlassPrecision, multiclass_precision), (MulticlassRecall, multiclass_recall)]
)
@pytest.mark.parametrize(
    ("k", "preds", "target", "average", "expected_prec", "expected_recall"),
    [
        (1, _mc_k_preds, _mc_k_target, "micro", torch.tensor(2 / 3), torch.tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", torch.tensor(1.0), torch.tensor(1.0)),
        (3, _mc_k_preds, _mc_k_target, "micro", torch.tensor(1.0), torch.tensor(1.0)),
        (1, _mc_k_preds2, _mc_k_targets2, "macro", torch.tensor(1 / 3), torch.tensor(1 / 2)),
        (2, _mc_k_preds2, _mc_k_targets2, "macro", torch.tensor(1 / 3), torch.tensor(1 / 2)),
        (3, _mc_k_preds2, _mc_k_targets2, "macro", torch.tensor(1.0), torch.tensor(1.0)),
        (1, _mc_k_preds3, _mc_k_target3, "macro", torch.tensor(0.1111), torch.tensor(0.3333)),
        (2, _mc_k_preds3, _mc_k_target3, "macro", torch.tensor(0.8333), torch.tensor(0.8333)),
        (3, _mc_k_preds3, _mc_k_target3, "macro", torch.tensor(1.0), torch.tensor(1.0)),
        (1, _mc_k_preds3, _mc_k_target3, "micro", torch.tensor(0.2500), torch.tensor(0.2500)),
        (2, _mc_k_preds3, _mc_k_target3, "micro", torch.tensor(0.7500), torch.tensor(0.7500)),
        (3, _mc_k_preds3, _mc_k_target3, "micro", torch.tensor(1.0), torch.tensor(1.0)),
        (1, _mc_k_preds3, _mc_k_target3, "weighted", torch.tensor(0.0833), torch.tensor(0.2500)),
        (2, _mc_k_preds3, _mc_k_target3, "weighted", torch.tensor(0.8750), torch.tensor(0.7500)),
        (3, _mc_k_preds3, _mc_k_target3, "weighted", torch.tensor(1.0), torch.tensor(1.0)),
        (1, _mc_k_preds3, _mc_k_target3, "none", torch.tensor([0.0000, 0.0000, 0.3333]), torch.tensor([0.0, 0.0, 1.0])),
        (2, _mc_k_preds3, _mc_k_target3, "none", torch.tensor([1.0000, 1.0000, 0.5000]), torch.tensor([0.5, 1.0, 1.0])),
        (3, _mc_k_preds3, _mc_k_target3, "none", torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])),
    ],
)
def test_top_k(
    metric_class,
    metric_fn,
    k: int,
    preds: Tensor,
    target: Tensor,
    average: str,
    expected_prec: Tensor,
    expected_recall: Tensor,
):
    """A test to validate top_k functionality for precision and recall."""
    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    result = expected_prec if metric_class.__name__ == "MulticlassPrecision" else expected_recall

    assert torch.allclose(class_metric.compute(), result, atol=1e-4, rtol=1e-4)
    assert torch.allclose(
        metric_fn(preds, target, top_k=k, average=average, num_classes=3), result, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("num_classes", [5])
def test_multiclass_precision_recall_with_top_k(num_classes):
    """Test that Precision and Recall increase monotonically with top_k and equal 1 when top_k equals num_classes.

    Args:
        num_classes: Number of classes in the classification task.

    The test verifies two properties:
    1. Precision and Recall increases or stays the same as top_k increases
    2. Precision and Recall equals 1 when top_k equals num_classes

    """
    preds = torch.randn(200, num_classes).softmax(dim=-1)
    target = torch.randint(num_classes, (200,))

    previous_precision = 0.0
    for k in range(1, num_classes + 1):
        precision_score = MulticlassPrecision(num_classes=num_classes, top_k=k, average="macro")
        precision = precision_score(preds, target)

        assert precision >= previous_precision, f"Precision did not increase for top_k={k}"
        previous_precision = precision

        if k == num_classes:
            assert torch.isclose(precision, torch.tensor(1.0)), (
                f"Precision is not 1 for top_k={k} when num_classes={num_classes}"
            )

    previous_recall = 0.0
    for k in range(1, num_classes + 1):
        recall_score = MulticlassRecall(num_classes=num_classes, top_k=k, average="macro")
        recall = recall_score(preds, target)

        assert recall >= previous_recall, f"Recall did not increase for top_k={k}"
        previous_recall = recall

        if k == num_classes:
            assert torch.isclose(recall, torch.tensor(1.0)), (
                f"Recall is not 1 for top_k={k} when num_classes={num_classes}"
            )


@pytest.mark.parametrize(("num_classes", "k"), [(5, 3), (10, 5)])
def test_multiclass_precision_recall_top_k_equivalence(num_classes, k):
    """Test that top-k Precision and Recall scores are equivalent to corrected top-1 scores."""
    preds = torch.randn(200, num_classes).softmax(dim=-1)
    target = torch.randint(num_classes, (200,))

    precision_top_k = MulticlassPrecision(num_classes=num_classes, top_k=k, average="macro")
    precision_top_1 = MulticlassPrecision(num_classes=num_classes, top_k=1, average="macro")

    recall_top_k = MulticlassRecall(num_classes=num_classes, top_k=k, average="macro")
    recall_top_1 = MulticlassRecall(num_classes=num_classes, top_k=1, average="macro")

    pred_top_k = torch.argsort(preds, dim=1, descending=True)[:, :k]
    pred_top_1 = pred_top_k[:, 0]
    target_in_top_k = (target.unsqueeze(1) == pred_top_k).any(dim=1)
    pred_corrected_top_k = torch.where(target_in_top_k, target, pred_top_1)

    precision_score_top_k = precision_top_k(preds, target)
    precision_score_corrected = precision_top_1(pred_corrected_top_k, target)

    recall_score_top_k = recall_top_k(preds, target)
    recall_score_corrected = recall_top_1(pred_corrected_top_k, target)

    assert torch.isclose(precision_score_top_k, precision_score_corrected), (
        f"Top-{k} Precision ({precision_score_top_k}) does not match "
        f"corrected top-1 Precision ({precision_score_corrected})"
    )

    assert torch.isclose(recall_score_top_k, recall_score_corrected), (
        f"Top-{k} Recall ({recall_score_top_k}) does not match corrected top-1 Recall ({recall_score_corrected})"
    )


def _reference_sklearn_precision_recall_multilabel_global(preds, target, sk_fn, ignore_index, average, zero_division):
    if average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
        return sk_fn(target, preds, zero_division=zero_division)

    precision_recall, weights = [], []
    for i in range(preds.shape[1]):
        pred, true = preds[:, i].flatten(), target[:, i].flatten()
        true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
        precision_recall.append(sk_fn(true, pred, zero_division=zero_division))
        confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
        weights.append(confmat[1, 1] + confmat[1, 0])
    res = np.stack(precision_recall, axis=0)

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


def _reference_sklearn_precision_recall_multilabel_local(preds, target, sk_fn, ignore_index, average, zero_division):
    precision_recall, weights = [], []
    for i in range(preds.shape[0]):
        if average == "micro":
            pred, true = preds[i].flatten(), target[i].flatten()
            true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
            precision_recall.append(sk_fn(true, pred, zero_division=zero_division))
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            weights.append(confmat[1, 1] + confmat[1, 0])
        else:
            scores, w = [], []
            for j in range(preds.shape[1]):
                pred, true = preds[i, j], target[i, j]
                true, pred = remove_ignore_index(target=true, preds=pred, ignore_index=ignore_index)
                scores.append(sk_fn(true, pred, zero_division=zero_division))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                w.append(confmat[1, 1] + confmat[1, 0])
            precision_recall.append(np.stack(scores))
            weights.append(np.stack(w))
    if average == "micro":
        return np.array(precision_recall)
    res = np.stack(precision_recall, 0)
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


def _reference_sklearn_precision_recall_multilabel(
    preds, target, sk_fn, ignore_index, multidim_average, average, zero_division=0, num_classes: int = NUM_CLASSES
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if ignore_index is None and multidim_average == "global":
        return sk_fn(
            target.transpose(0, 2, 1).reshape(-1, num_classes),
            preds.transpose(0, 2, 1).reshape(-1, num_classes),
            average=average,
            zero_division=zero_division,
        )
    if multidim_average == "global":
        return _reference_sklearn_precision_recall_multilabel_global(
            preds, target, sk_fn, ignore_index, average, zero_division
        )
    return _reference_sklearn_precision_recall_multilabel_local(
        preds, target, sk_fn, ignore_index, average, zero_division
    )


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (MultilabelPrecision, multilabel_precision, sk_precision_score),
        (MultilabelRecall, multilabel_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestMultilabelPrecisionRecall(MetricTester):
    """Test class for `MultilabelPrecisionRecall` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_precision_recall(
        self, ddp, inputs, module, functional, compare, ignore_index, multidim_average, average, zero_division
    ):
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_precision_recall_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_precision_recall_functional(
        self, inputs, module, functional, compare, ignore_index, multidim_average, average, zero_division
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_precision_recall_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "zero_division": zero_division,
            },
        )

    def test_multilabel_precision_recall_differentiability(self, inputs, module, functional, compare):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_half_cpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_corner_case():
    """Issue: https://github.com/Lightning-AI/torchmetrics/issues/1692."""
    # simulate the output of a perfect predictor (i.e. preds == target)
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = target.clone()

    metric = MulticlassPrecision(num_classes=3, average="none", ignore_index=0)
    res = metric(preds, target)
    assert torch.allclose(res, torch.tensor([0.0, 1.0, 1.0]))

    metric = MulticlassRecall(num_classes=3, average="none", ignore_index=0)
    res = metric(preds, target)
    assert torch.allclose(res, torch.tensor([0.0, 1.0, 1.0]))

    metric = MulticlassPrecision(num_classes=3, average="macro", ignore_index=0)
    res = metric(preds, target)
    assert res == 1.0

    metric = MulticlassRecall(num_classes=3, average="macro", ignore_index=0)
    res = metric(preds, target)
    assert res == 1.0


@pytest.mark.parametrize(
    ("metric", "kwargs", "base_metric"),
    [
        (BinaryPrecision, {"task": "binary"}, Precision),
        (MulticlassPrecision, {"task": "multiclass", "num_classes": 3}, Precision),
        (MultilabelPrecision, {"task": "multilabel", "num_labels": 3}, Precision),
        (None, {"task": "not_valid_task"}, Precision),
        (BinaryRecall, {"task": "binary"}, Recall),
        (MulticlassRecall, {"task": "multiclass", "num_classes": 3}, Recall),
        (MultilabelRecall, {"task": "multilabel", "num_labels": 3}, Recall),
        (None, {"task": "not_valid_task"}, Recall),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
