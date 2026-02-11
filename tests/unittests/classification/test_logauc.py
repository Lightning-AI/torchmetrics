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
from sklearn.metrics import auc, roc_curve

from torchmetrics.classification.logauc import BinaryLogAUC, LogAUC, MulticlassLogAUC, MultilabelLogAUC
from torchmetrics.functional.classification.logauc import binary_logauc, multiclass_logauc, multilabel_logauc
from torchmetrics.functional.classification.roc import binary_roc
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases


def range_logauc(
    true_y: np.ndarray, predicted_score: np.ndarray, fpr_range: tuple[float, float] = (0.001, 0.1)
) -> float:
    """Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).

    Adapted from Therapeutics Commons (TDC): Multimodal Foundation for Therapeutic Science
    https://github.com/mims-harvard/TDC/blob/main/tdc/evaluator.py#L66

    Original Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)

    This metric is used in applications where the positive and negative are imbalanced and a low
    false positive rate is of high importance. The score is computed by first computing the ROC curve,
    which then is interpolated to the specified range of false positive rates (FPR) and then the log
    is taken of the FPR before the area under the curve (AUC) is computed.

    A perfect classifier gets a logAUC[0.001, 0.1] of 1, while a random classifier gets a
    logAUC[0.001, 0.1] of around 0.0215.

    References:
        [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
        Desolvation in Molecular Docking. Journal of Chemical Information and
        Modeling, 2010. 50(9): p. 1561-1573.
        [2] Liu, Yunchao, et al. "Interpretable Chirality-Aware Graph Neural
        Network for Quantitative Structure Activity Relationship Modeling in
        Drug Discovery." bioRxiv (2022).

    Args:
        true_y: numpy array of the ground truth. Values are either 0 (inactive) or 1(active).
        predicted_score: numpy array of the predicted score (The score does not have to be between 0 and 1)
        fpr_range: the range for calculating the logAUC formatted in (x, y) with x being the lower bound
            and y being the upper bound

    Returns:
        float: the logAUC score

    """
    # FPR range validity check
    if fpr_range is None:
        raise ValueError("FPR range cannot be None")
    lower_bound = fpr_range[0]
    upper_bound = fpr_range[1]
    if lower_bound >= upper_bound:
        raise ValueError("FPR upper_bound must be greater than lower_bound")

    fpr, tpr, _thresholds = roc_curve(true_y, predicted_score, pos_label=1)

    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], fpr, tpr))
    fpr = np.append(fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is the y coordinate.
    log_fpr = np.log10(fpr)
    x = log_fpr
    y = tpr
    lower_bound = np.log10(lower_bound)
    upper_bound = np.log10(upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx : upper_bound_idx + 1]
    trim_y = y[lower_bound_idx : upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / (upper_bound - lower_bound)
    return float(area)


seed_all(42)


def _binary_compare_implementation(preds, target, fpr_range, ignore_index=None):
    """Binary comparison function for logauc."""
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return range_logauc(target, preds, fpr_range=fpr_range)


@pytest.mark.parametrize("inputs", [_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]])
class TestBinaryLogAUC(MetricTester):
    """Test class for `BinaryLogAUC` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    def test_binary_logauc(self, inputs, ddp, fpr_range):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryLogAUC,
            reference_metric=partial(_binary_compare_implementation, fpr_range=fpr_range),
            metric_args={
                "fpr_range": fpr_range,
                "thresholds": None,
            },
        )

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_logauc_functional(self, inputs, fpr_range, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_logauc,
            reference_metric=partial(_binary_compare_implementation, fpr_range=fpr_range, ignore_index=ignore_index),
            metric_args={
                "fpr_range": fpr_range,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_logauc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_logauc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_logauc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_logauc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_logauc(pred, true, thresholds=None)
            ap2 = binary_logauc(pred, true, thresholds=threshold_fn(t.flip(0)))
            assert torch.allclose(ap1, ap2, atol=self.atol)


def _multiclass_compare_implementation(preds, target, fpr_range, average):
    """Multiclass comparison function for logauc."""
    preds = preds.permute(0, 2, 1).reshape(-1, NUM_CLASSES).numpy() if preds.ndim == 3 else preds.numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)

    scores = []
    for i in range(NUM_CLASSES):
        p, t = preds[:, i], (target == i).astype(int)
        scores.append(range_logauc(t, p, fpr_range=fpr_range))
    if average == "macro":
        return np.mean(scores)
    return scores


@pytest.mark.parametrize(
    "inputs", [_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5]]
)
class TestMulticlassLogAUC(MetricTester):
    """Test class for `MulticlassLogAUC` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("average", ["macro", None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_logauc(self, inputs, fpr_range, average, ddp):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassLogAUC,
            reference_metric=partial(_multiclass_compare_implementation, fpr_range=fpr_range, average=average),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "fpr_range": fpr_range,
                "average": average,
            },
        )

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("average", ["macro", None])
    def test_multiclass_logauc_functional(self, inputs, fpr_range, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_logauc,
            reference_metric=partial(_multiclass_compare_implementation, fpr_range=fpr_range, average=average),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "fpr_range": fpr_range,
                "average": average,
            },
        )

    def test_multiclass_logauc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassLogAUC,
            metric_functional=multiclass_logauc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_logauc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassLogAUC,
            metric_functional=multiclass_logauc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_logauc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassLogAUC,
            metric_functional=multiclass_logauc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    def test_multiclass_logauc_threshold_arg(self, inputs):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 2)) + 1e-6  # rounding will simulate binning
            ap1 = multiclass_logauc(pred, true, num_classes=NUM_CLASSES, average="macro", thresholds=None)
            ap2 = multiclass_logauc(
                pred, true, num_classes=NUM_CLASSES, average="macro", thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2, atol=self.atol)


def _multilabel_compare_implementation(preds, target, fpr_range, average):
    if preds.ndim > 2:
        target = target.transpose(2, 1).reshape(-1, NUM_CLASSES)
        preds = preds.transpose(2, 1).reshape(-1, NUM_CLASSES)
    target = target.numpy()
    preds = preds.numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    scores = []
    for i in range(NUM_CLASSES):
        p, t = preds[:, i], target[:, i]
        scores.append(range_logauc(t, p, fpr_range=fpr_range))
    if average == "macro":
        return np.mean(scores)
    return scores


@pytest.mark.parametrize(
    "inputs", [_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5]]
)
class TestMultilabelLogAUC(MetricTester):
    """Test class for `MultilabelLogAUC` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("average", ["macro", None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_logauc(self, inputs, ddp, fpr_range, average):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelLogAUC,
            reference_metric=partial(_multilabel_compare_implementation, fpr_range=fpr_range, average=average),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "fpr_range": fpr_range,
            },
        )

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("average", ["macro", None])
    def test_multilabel_logauc_functional(self, inputs, fpr_range, average):
        """Test functional implementation of metric."""
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_logauc,
            reference_metric=partial(_multilabel_compare_implementation, fpr_range=fpr_range, average=average),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "fpr_range": fpr_range,
            },
        )

    def test_multiclass_logauc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelLogAUC,
            metric_functional=multilabel_logauc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_logauc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelLogAUC,
            metric_functional=multilabel_logauc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_logauc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelLogAUC,
            metric_functional=multilabel_logauc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    def test_multilabel_logauc_threshold_arg(self, inputs):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            ap1 = multilabel_logauc(pred, true, num_labels=NUM_CLASSES, average="macro", thresholds=None)
            ap2 = multilabel_logauc(
                pred, true, num_labels=NUM_CLASSES, average="macro", thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2, atol=self.atol)


@pytest.mark.parametrize(
    "metric",
    [
        BinaryLogAUC,
        partial(MulticlassLogAUC, num_classes=NUM_CLASSES),
        partial(MultilabelLogAUC, num_labels=NUM_CLASSES),
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
        (BinaryLogAUC, {"task": "binary"}),
        (MulticlassLogAUC, {"task": "multiclass", "num_classes": 3}),
        (MultilabelLogAUC, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=LogAUC):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
