# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report as sk_classification_report
from torch import Tensor, tensor

from torchmetrics.classification.classification_report import (
    BinaryClassificationReport,
    ClassificationReport,
    MulticlassClassificationReport,
    MultilabelClassificationReport,
)
from torchmetrics.functional.classification.classification_report import (
    binary_classification_report,
    multiclass_classification_report,
    multilabel_classification_report,
)
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import inject_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _to_numpy(x):
    """Convert to numpy."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _compare_dicts(tm_result, ref_result, atol=1e-4):
    """Compare two classification report dicts."""
    for key in ref_result:
        if key not in tm_result:
            continue
        ref_val = ref_result[key]
        tm_val = tm_result[key]
        if isinstance(ref_val, dict):
            _compare_dicts(tm_val, ref_val, atol=atol)
        else:
            rv = float(_to_numpy(ref_val)) if np.ndim(_to_numpy(ref_val)) == 0 else _to_numpy(ref_val)
            tv = float(_to_numpy(tm_val)) if np.ndim(_to_numpy(tm_val)) == 0 else _to_numpy(tm_val)
            if isinstance(rv, (int, float)) and isinstance(tv, (int, float)):
                assert np.isclose(tv, rv, atol=atol), f"Mismatch at {key}: tm={tv}, ref={rv}"


def _is_multidim(preds, target):
    """Check if inputs have extra dimensions beyond batch and class dims."""
    # For binary: multi-dim if preds.ndim > 2 (B, N, D...)
    # For multiclass: multi-dim if target.ndim > 2 (B, N, D...) or preds.ndim > 3 (B, N, C, D...)
    return target.ndim > 2


def _reference_sklearn_binary(preds, target, ignore_index=None, zero_division=0):
    """Reference implementation using sklearn for binary classification report."""
    preds_np = _to_numpy(preds).flatten()
    target_np = _to_numpy(target).flatten()
    if np.issubdtype(preds_np.dtype, np.floating):
        if not ((preds_np > 0) & (preds_np < 1)).all():
            preds_np = sigmoid(preds_np)
        preds_np = (preds_np >= THRESHOLD).astype(np.uint8)
    if ignore_index is not None:
        mask = target_np != ignore_index
        target_np = target_np[mask]
        preds_np = preds_np[mask]
    report = sk_classification_report(target_np, preds_np, output_dict=True, zero_division=zero_division)
    return {
        "0": {
            "precision": report["0"]["precision"],
            "recall": report["0"]["recall"],
            "f1_score": report["0"]["f1-score"],
            "support": report["0"]["support"],
        },
        "1": {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "support": report["1"]["support"],
        },
        "macro": {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "support": report["macro avg"]["support"],
        },
        "weighted": {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "support": report["weighted avg"]["support"],
        },
    }


def _reference_sklearn_multiclass(preds, target, ignore_index=None, zero_division=0):
    """Reference implementation using sklearn for multiclass classification report.

    Handles both single-dim and multi-dim inputs. For multi-dim, flattens
    the extra dimensions along the batch dimension to match torchmetrics
    ``multidim_average='global'`` behavior.
    """
    preds_np = _to_numpy(preds)
    target_np = _to_numpy(target)

    # If preds has extra class dim, convert to labels via argmax
    if preds_np.ndim == target_np.ndim + 1:
        # Class dim is at index 1 (after flattening B*N)
        preds_np = preds_np.argmax(axis=1)

    # Flatten all dims to 1D
    preds_flat = preds_np.flatten()
    target_flat = target_np.flatten()

    if ignore_index is not None:
        mask = target_flat != ignore_index
        target_flat = target_flat[mask]
        preds_flat = preds_flat[mask]

    report = sk_classification_report(
        target_flat, preds_flat, output_dict=True, zero_division=zero_division,
        labels=list(range(NUM_CLASSES))
    )
    result = {}
    for c in range(NUM_CLASSES):
        key = str(c)
        if key in report:
            result[key] = {
                "precision": report[key]["precision"],
                "recall": report[key]["recall"],
                "f1_score": report[key]["f1-score"],
                "support": report[key]["support"],
            }
        else:
            result[key] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    total_support = sum(result[str(c)]["support"] for c in range(NUM_CLASSES))
    # For multiclass, micro avg = accuracy
    accuracy = report.get("accuracy", 0.0)
    result["micro"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1_score": accuracy,
        "support": total_support,
    }
    result["macro"] = {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"],
        "support": report["macro avg"]["support"],
    }
    result["weighted"] = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "support": report["weighted avg"]["support"],
    }
    return result


def _reference_sklearn_multilabel(preds, target, zero_division=0):
    """Reference implementation using sklearn for multilabel classification report.

    Expects data already in torchmetrics format: (N, C, ...) where C = num_labels.
    For single-dim: (N, C) -> use sklearn directly.
    For multi-dim: (N, C, D) -> flatten each label's D values independently.
    """
    preds_np = _to_numpy(preds)
    target_np = _to_numpy(target)

    if np.issubdtype(preds_np.dtype, np.floating):
        if not ((preds_np > 0) & (preds_np < 1)).all():
            preds_np = sigmoid(preds_np)
        preds_np = (preds_np >= THRESHOLD).astype(np.uint8)

    n_labels = preds_np.shape[1] if preds_np.ndim >= 2 else NUM_CLASSES

    # Single-dim case: (N, C)
    if preds_np.ndim == 2:
        report = sk_classification_report(
            target_np, preds_np, output_dict=True, zero_division=zero_division
        )
        result = {}
        for lbl_idx in range(n_labels):
            key = f"label_{lbl_idx}"
            sk_key = str(lbl_idx)
            if sk_key in report:
                result[key] = {
                    "precision": report[sk_key]["precision"],
                    "recall": report[sk_key]["recall"],
                    "f1_score": report[sk_key]["f1-score"],
                    "support": report[sk_key]["support"],
                }
            else:
                result[key] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

        total_support = sum(
            report[str(lbl_idx)]["support"] for lbl_idx in range(n_labels) if str(lbl_idx) in report
        )
        result["micro"] = {
            "precision": report.get("micro avg", {}).get("precision", 0.0),
            "recall": report.get("micro avg", {}).get("recall", 0.0),
            "f1_score": report.get("micro avg", {}).get("f1-score", 0.0),
            "support": report.get("micro avg", {}).get("support", total_support),
        }
        result["macro"] = {
            "precision": report.get("macro avg", {}).get("precision", 0.0),
            "recall": report.get("macro avg", {}).get("recall", 0.0),
            "f1_score": report.get("macro avg", {}).get("f1-score", 0.0),
            "support": report.get("macro avg", {}).get("support", total_support),
        }
        result["weighted"] = {
            "precision": report.get("weighted avg", {}).get("precision", 0.0),
            "recall": report.get("weighted avg", {}).get("recall", 0.0),
            "f1_score": report.get("weighted avg", {}).get("f1-score", 0.0),
            "support": report.get("weighted avg", {}).get("support", total_support),
        }
        return result

    # Multi-dim case: (N, C, D) - flatten D along batch for each label independently
    # This matches torchmetrics multidim_average='global' behavior
    per_label_stats = []
    for lbl_idx in range(n_labels):
        p_l = preds_np[:, lbl_idx, :].flatten()
        t_l = target_np[:, lbl_idx, :].flatten()
        tp = float(((p_l == 1) & (t_l == 1)).sum())
        fp = float(((p_l == 1) & (t_l == 0)).sum())
        fn = float(((p_l == 0) & (t_l == 1)).sum())
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
        rec = tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float(zero_division)
        per_label_stats.append({
            "precision": prec, "recall": rec, "f1_score": f1, "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        })

    result = {}
    for lbl_idx in range(n_labels):
        result[f"label_{lbl_idx}"] = {
            "precision": per_label_stats[lbl_idx]["precision"],
            "recall": per_label_stats[lbl_idx]["recall"],
            "f1_score": per_label_stats[lbl_idx]["f1_score"],
            "support": per_label_stats[lbl_idx]["support"],
        }

    tp_sum = sum(s["tp"] for s in per_label_stats)
    fp_sum = sum(s["fp"] for s in per_label_stats)
    fn_sum = sum(s["fn"] for s in per_label_stats)
    total_support = sum(s["support"] for s in per_label_stats)

    micro_prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else float(zero_division)
    micro_rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else float(zero_division)
    micro_f1 = (
        2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)
        if (2 * tp_sum + fp_sum + fn_sum) > 0
        else float(zero_division)
    )

    result["micro"] = {
        "precision": micro_prec, "recall": micro_rec, "f1_score": micro_f1,
        "support": total_support,
    }

    macro_prec = np.mean([s["precision"] for s in per_label_stats])
    macro_rec = np.mean([s["recall"] for s in per_label_stats])
    macro_f1 = np.mean([s["f1_score"] for s in per_label_stats])

    result["macro"] = {
        "precision": macro_prec, "recall": macro_rec, "f1_score": macro_f1,
        "support": total_support,
    }

    if total_support > 0:
        weights = np.array([s["support"] for s in per_label_stats])
        weighted_prec = np.average([s["precision"] for s in per_label_stats], weights=weights)
        weighted_rec = np.average([s["recall"] for s in per_label_stats], weights=weights)
        weighted_f1 = np.average([s["f1_score"] for s in per_label_stats], weights=weights)
    else:
        weighted_prec = weighted_rec = weighted_f1 = 0.0

    result["weighted"] = {
        "precision": weighted_prec, "recall": weighted_rec, "f1_score": weighted_f1,
        "support": total_support,
    }

    return result


def _flatten_binary(preds, target):
    """Flatten binary test inputs to single batch (N,)."""
    return preds.reshape(-1), target.reshape(-1)


def _flatten_multiclass(preds, target):
    """Flatten multiclass test inputs to proper torchmetrics format.

    torchmetrics multiclass expects:
    - Label preds/target: same shape, with multidim averaging handled internally
    - Prob/logit preds: one extra dim for class probabilities

    Test input shapes (B=4, N=32, C=5, D=3):
    - single_dim labels: preds (B, N) target (B, N)
    - single_dim probs: preds (B, N, C) target (B, N)
    - single_dim logits: preds (B, N, C) target (B, N)
    - multi_dim labels: preds (B, N, D) target (B, N, D)
    - multi_dim probs: preds (B, N, C, D) target (B, N, D)
    - multi_dim logits: preds (B, N, C, D) target (B, N, D)

    Flatten B+N dims together to get (N, ...) format for torchmetrics.
    """
    is_prob = preds.ndim == target.ndim + 1
    if is_prob:
        preds_flat = preds.reshape(-1, *preds.shape[2:])
        target_flat = target.reshape(-1, *target.shape[2:])
        return preds_flat, target_flat
    preds_flat = preds.reshape(-1, *preds.shape[2:])
    target_flat = target.reshape(-1, *target.shape[2:])
    return preds_flat, target_flat


def _flatten_multilabel(preds, target):
    """Reshape multilabel test inputs to proper torchmetrics format.

    torchmetrics multilabel expects: (N, C, ...) where C = num_labels.
    Test input shapes use (B, N, C) convention, so we need to permute
    to (B, C, N) for torchmetrics, then flatten B*N together.
    """
    nc = NUM_CLASSES
    if preds.ndim == 3 and preds.shape[-1] == nc:
        # (B, N, C) -> (B*C, N) = (N, C) after flatten
        return preds.reshape(-1, nc), target.reshape(-1, nc)
    if preds.ndim == 4 and preds.shape[-2] == nc:
        # (B, N, C, D) -> (B, C, N, D) -> (B*C, N, D) for torchmetrics
        preds_tm = preds.permute(0, 2, 1, 3).reshape(-1, nc, preds.shape[-1])
        target_tm = target.permute(0, 2, 1, 3).reshape(-1, nc, target.shape[-1])
        return preds_tm, target_tm
    return preds, target


# =====================================================================
# Binary tests
# =====================================================================


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryClassificationReport:
    """Test class for `BinaryClassificationReport` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_classification_report_class(self, inputs, ignore_index, zero_division):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        preds_flat, target_flat = _flatten_binary(preds, target)

        metric = BinaryClassificationReport(
            threshold=THRESHOLD,
            multidim_average="global",
            ignore_index=ignore_index,
            zero_division=zero_division,
        )
        tm_result = metric(preds_flat, target_flat)
        ref_result = _reference_sklearn_binary(
            preds_flat, target_flat, ignore_index=ignore_index, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_classification_report_functional(self, inputs, ignore_index, zero_division):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        preds_flat, target_flat = _flatten_binary(preds, target)

        tm_result = binary_classification_report(
            preds_flat, target_flat,
            threshold=THRESHOLD,
            multidim_average="global",
            ignore_index=ignore_index,
            zero_division=zero_division,
        )
        ref_result = _reference_sklearn_binary(
            preds_flat, target_flat, ignore_index=ignore_index, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)


# =====================================================================
# Multiclass tests
# =====================================================================


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassClassificationReport:
    """Test class for `MulticlassClassificationReport` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_classification_report_class(self, inputs, ignore_index, zero_division):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        preds_flat, target_flat = _flatten_multiclass(preds, target)

        metric = MulticlassClassificationReport(
            num_classes=NUM_CLASSES,
            multidim_average="global",
            ignore_index=ignore_index,
            zero_division=zero_division,
        )
        tm_result = metric(preds_flat, target_flat)
        ref_result = _reference_sklearn_multiclass(
            preds_flat, target_flat, ignore_index=ignore_index, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_classification_report_functional(self, inputs, ignore_index, zero_division):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        preds_flat, target_flat = _flatten_multiclass(preds, target)

        tm_result = multiclass_classification_report(
            preds_flat, target_flat, num_classes=NUM_CLASSES,
            multidim_average="global",
            ignore_index=ignore_index,
            zero_division=zero_division,
        )
        ref_result = _reference_sklearn_multiclass(
            preds_flat, target_flat, ignore_index=ignore_index, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)


# =====================================================================
# Multilabel tests
# =====================================================================


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelClassificationReport:
    """Test class for `MultilabelClassificationReport` metric."""

    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_classification_report_class(self, inputs, zero_division):
        """Test class implementation of metric."""
        preds_orig, target_orig = inputs
        preds_tm, target_tm = _flatten_multilabel(preds_orig, target_orig)

        metric = MultilabelClassificationReport(
            num_labels=NUM_CLASSES,
            threshold=THRESHOLD,
            multidim_average="global",
            ignore_index=None,
            zero_division=zero_division,
        )
        tm_result = metric(preds_tm, target_tm)
        # Reference uses torchmetrics-format data
        ref_result = _reference_sklearn_multilabel(
            preds_tm, target_tm, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)

    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_classification_report_functional(self, inputs, zero_division):
        """Test functional implementation of metric."""
        preds_orig, target_orig = inputs
        preds_tm, target_tm = _flatten_multilabel(preds_orig, target_orig)

        tm_result = multilabel_classification_report(
            preds_tm, target_tm, num_labels=NUM_CLASSES,
            threshold=THRESHOLD,
            multidim_average="global",
            ignore_index=None,
            zero_division=zero_division,
        )
        ref_result = _reference_sklearn_multilabel(
            preds_tm, target_tm, zero_division=zero_division
        )
        _compare_dicts(tm_result, ref_result)


# =====================================================================
# Wrapper tests
# =====================================================================


def test_classification_report_wrapper_binary():
    """Test the ClassificationReport wrapper for binary task."""
    target = tensor([0, 1, 0, 1, 0, 1])
    preds = tensor([0, 0, 1, 1, 0, 1])

    metric = ClassificationReport(task="binary")
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert "macro" in report
    assert "weighted" in report

    metric2 = BinaryClassificationReport()
    report2 = metric2(preds, target)
    for key in report:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(report[key][metric_key], report2[key][metric_key], atol=1e-8)


def test_classification_report_wrapper_multiclass():
    """Test the ClassificationReport wrapper for multiclass task."""
    target = tensor([2, 1, 0, 0])
    preds = tensor([2, 1, 0, 1])

    metric = ClassificationReport(task="multiclass", num_classes=3)
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert "2" in report
    assert "micro" in report
    assert "macro" in report
    assert "weighted" in report

    metric2 = MulticlassClassificationReport(num_classes=3)
    report2 = metric2(preds, target)
    for key in report:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(report[key][metric_key], report2[key][metric_key], atol=1e-8)


def test_classification_report_wrapper_multilabel():
    """Test the ClassificationReport wrapper for multilabel task."""
    target = tensor([[0, 1, 0], [1, 0, 1]])
    preds = tensor([[0, 0, 1], [1, 0, 1]])

    metric = ClassificationReport(task="multilabel", num_labels=3)
    report = metric(preds, target)
    assert "label_0" in report
    assert "label_1" in report
    assert "label_2" in report
    assert "micro" in report
    assert "macro" in report
    assert "weighted" in report


def test_classification_report_wrapper_errors():
    """Test that ClassificationReport wrapper raises errors for missing arguments."""
    with pytest.raises(ValueError, match="`num_classes` is required"):
        ClassificationReport(task="multiclass")

    with pytest.raises(ValueError, match="`num_labels` is required"):
        ClassificationReport(task="multilabel")


# =====================================================================
# Edge case tests
# =====================================================================


def test_binary_all_wrong():
    """Test binary classification report when all predictions are wrong."""
    target = tensor([0, 0, 0, 1, 1, 1])
    preds = tensor([1, 1, 1, 0, 0, 0])

    metric = BinaryClassificationReport()
    report = metric(preds, target)
    assert report["0"]["precision"].item() == 0.0
    assert report["0"]["recall"].item() == 0.0
    assert report["1"]["precision"].item() == 0.0
    assert report["1"]["recall"].item() == 0.0


def test_binary_all_correct():
    """Test binary classification report when all predictions are correct."""
    target = tensor([0, 0, 0, 1, 1, 1])
    preds = tensor([0, 0, 0, 1, 1, 1])

    metric = BinaryClassificationReport()
    report = metric(preds, target)
    assert report["0"]["precision"].item() == 1.0
    assert report["0"]["recall"].item() == 1.0
    assert report["1"]["precision"].item() == 1.0
    assert report["1"]["recall"].item() == 1.0
    assert report["macro"]["f1_score"].item() == 1.0


def test_multiclass_float_preds():
    """Test multiclass classification report with float predictions."""
    target = tensor([2, 1, 0, 0])
    preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])

    metric = MulticlassClassificationReport(num_classes=3)
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert "2" in report


def test_multiclass_zero_division():
    """Test multiclass classification report with zero division."""
    target = tensor([0, 1, 0, 1])
    preds = tensor([0, 0, 0, 0])

    metric = MulticlassClassificationReport(num_classes=3, zero_division=0.0)
    report = metric(preds, target)
    assert report["2"]["precision"].item() == 0.0
    assert report["2"]["recall"].item() == 0.0

    metric1 = MulticlassClassificationReport(num_classes=3, zero_division=1.0)
    report1 = metric1(preds, target)
    assert report1["2"]["precision"].item() == 1.0
    assert report1["2"]["recall"].item() == 1.0


def test_binary_float_preds():
    """Test binary classification report with float predictions."""
    target = tensor([0, 1, 0, 1, 0, 1])
    preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])

    metric = BinaryClassificationReport()
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert report["macro"]["f1_score"].item() > 0.0


def test_multiclass_incremental():
    """Test that incremental updates give the same result as batch."""
    target = tensor([2, 1, 0, 0, 1, 2])
    preds = tensor([2, 1, 0, 1, 1, 0])

    metric_batch = MulticlassClassificationReport(num_classes=3)
    report_batch = metric_batch(preds, target)

    metric_incr = MulticlassClassificationReport(num_classes=3)
    metric_incr.update(preds[:3], target[:3])
    metric_incr.update(preds[3:], target[3:])
    report_incr = metric_incr.compute()

    for key in report_batch:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                report_batch[key][metric_key], report_incr[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"


def test_multilabel_incremental():
    """Test that incremental updates give the same result as batch for multilabel."""
    target = tensor([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]])
    preds = tensor([[0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]])

    metric_batch = MultilabelClassificationReport(num_labels=3)
    report_batch = metric_batch(preds, target)

    metric_incr = MultilabelClassificationReport(num_labels=3)
    metric_incr.update(preds[:2], target[:2])
    metric_incr.update(preds[2:], target[2:])
    report_incr = metric_incr.compute()

    for key in report_batch:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                report_batch[key][metric_key], report_incr[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"


def test_multiclass_sklearn_match():
    """Test multiclass report against sklearn for a known example."""
    target = tensor([0, 1, 2, 2, 0, 1, 0, 2, 1, 1])
    preds = tensor([0, 2, 1, 2, 0, 0, 1, 2, 1, 0])

    metric = MulticlassClassificationReport(num_classes=3)
    report = metric(preds, target)

    sk_report = sk_classification_report(target.numpy(), preds.numpy(), output_dict=True, zero_division=0)

    for c in range(3):
        key = str(c)
        assert np.isclose(report[key]["precision"].item(), sk_report[key]["precision"], atol=1e-4)
        assert np.isclose(report[key]["recall"].item(), sk_report[key]["recall"], atol=1e-4)
        assert np.isclose(report[key]["f1_score"].item(), sk_report[key]["f1-score"], atol=1e-4)
        assert np.isclose(report[key]["support"].item(), sk_report[key]["support"], atol=1e-4)

    assert np.isclose(report["macro"]["precision"].item(), sk_report["macro avg"]["precision"], atol=1e-4)
    assert np.isclose(report["macro"]["recall"].item(), sk_report["macro avg"]["recall"], atol=1e-4)
    assert np.isclose(report["macro"]["f1_score"].item(), sk_report["macro avg"]["f1-score"], atol=1e-4)

    assert np.isclose(report["weighted"]["precision"].item(), sk_report["weighted avg"]["precision"], atol=1e-4)
    assert np.isclose(report["weighted"]["recall"].item(), sk_report["weighted avg"]["recall"], atol=1e-4)
    assert np.isclose(report["weighted"]["f1_score"].item(), sk_report["weighted avg"]["f1-score"], atol=1e-4)


def test_multiclass_ignore_index():
    """Test multiclass classification report with ignore_index."""
    target = tensor([0, 1, 2, -1, 0, 1])
    preds = tensor([0, 2, 1, 2, 0, 0])

    metric = MulticlassClassificationReport(num_classes=3, ignore_index=-1)
    report = metric(preds, target)

    target_clean = tensor([0, 1, 2, 0, 1])
    preds_clean = tensor([0, 2, 1, 0, 0])

    metric_clean = MulticlassClassificationReport(num_classes=3)
    report_clean = metric_clean(preds_clean, target_clean)

    for key in report:
        for metric_key in ["precision", "recall", "f1_score"]:
            assert torch.allclose(
                report[key][metric_key], report_clean[key][metric_key], atol=1e-4,
            ), f"Mismatch at {key}.{metric_key}: {report[key][metric_key]} vs {report_clean[key][metric_key]}"


def test_binary_samplewise_runs():
    """Test that binary classification report samplewise mode runs without error."""
    target = torch.randint(0, 2, (8, 32, 3))
    preds = torch.rand(8, 32, 3)

    metric = BinaryClassificationReport(multidim_average="samplewise")
    result = metric(preds, target)
    assert "0" in result
    assert "1" in result
    assert "macro" in result
    assert "weighted" in result
    assert result["0"]["precision"].shape == (8,)


def test_multiclass_samplewise_runs():
    """Test that multiclass classification report samplewise mode runs without error."""
    target = torch.randint(0, 5, (8, 32, 3))
    preds = torch.randint(0, 5, (8, 32, 3))

    metric = MulticlassClassificationReport(num_classes=5, multidim_average="samplewise")
    result = metric(preds, target)
    assert "0" in result
    assert "4" in result
    assert "macro" in result


def test_multilabel_samplewise_runs():
    """Test that multilabel classification report samplewise mode runs without error."""
    target = torch.randint(0, 2, (8, 5, 32, 3))
    preds = torch.rand(8, 5, 32, 3)

    metric = MultilabelClassificationReport(num_labels=5, multidim_average="samplewise")
    result = metric(preds, target)
    assert "label_0" in result
    assert "label_4" in result
    assert "macro" in result


def test_binary_zero_division():
    """Test binary classification report with zero_division parameter."""
    target = tensor([1, 1, 1])
    preds = tensor([1, 1, 1])

    metric0 = BinaryClassificationReport(zero_division=0.0)
    report0 = metric0(preds, target)
    assert report0["0"]["precision"].item() == 0.0

    metric1 = BinaryClassificationReport(zero_division=1.0)
    report1 = metric1(preds, target)
    assert report1["0"]["precision"].item() == 1.0


def test_multiclass_multidim_global():
    """Test multiclass classification report with multi-dim inputs and global averaging."""
    target = torch.randint(0, 3, (32, 5))
    preds = torch.randint(0, 3, (32, 5))

    metric = MulticlassClassificationReport(num_classes=3, multidim_average="global")
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert "2" in report
    assert "micro" in report
    assert "macro" in report
    assert "weighted" in report


def test_multilabel_zero_division():
    """Test multilabel classification report with zero_division parameter."""
    # All predictions are 0, so label_1 has no positive predictions
    target = tensor([[0, 0], [0, 0]])
    preds = tensor([[0, 0], [0, 0]])

    metric0 = MultilabelClassificationReport(num_labels=2, zero_division=0.0)
    report0 = metric0(preds, target)
    assert report0["label_0"]["precision"].item() == 0.0

    metric1 = MultilabelClassificationReport(num_labels=2, zero_division=1.0)
    report1 = metric1(preds, target)
    assert report1["label_0"]["precision"].item() == 1.0


def test_binary_functional_vs_class():
    """Test that functional and class implementations give same results for binary."""
    target = tensor([0, 1, 0, 1, 0, 1])
    preds = tensor([0, 0, 1, 1, 0, 1])

    metric = BinaryClassificationReport()
    class_result = metric(preds, target)

    func_result = binary_classification_report(preds, target)

    for key in class_result:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                class_result[key][metric_key], func_result[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"


def test_multiclass_functional_vs_class():
    """Test that functional and class implementations give same results for multiclass."""
    target = tensor([2, 1, 0, 0])
    preds = tensor([2, 1, 0, 1])

    metric = MulticlassClassificationReport(num_classes=3)
    class_result = metric(preds, target)

    func_result = multiclass_classification_report(preds, target, num_classes=3)

    for key in class_result:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                class_result[key][metric_key], func_result[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"


def test_multilabel_functional_vs_class():
    """Test that functional and class implementations give same results for multilabel."""
    target = tensor([[0, 1, 0], [1, 0, 1]])
    preds = tensor([[0, 0, 1], [1, 0, 1]])

    metric = MultilabelClassificationReport(num_labels=3)
    class_result = metric(preds, target)

    func_result = multilabel_classification_report(preds, target, num_labels=3)

    for key in class_result:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                class_result[key][metric_key], func_result[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"


def test_binary_multidim_logits():
    """Test binary classification report with multi-dim logits."""
    target = torch.randint(0, 2, (10, 5))
    preds = torch.randn(10, 5)

    metric = BinaryClassificationReport(multidim_average="global")
    report = metric(preds, target)
    assert "0" in report
    assert "1" in report
    assert "macro" in report
    assert "weighted" in report


def test_multiclass_multidim_probs():
    """Test multiclass with multi-dim probability inputs."""
    target = torch.randint(0, 5, (8, 4))
    preds = torch.randn(8, 5, 4).softmax(dim=1)

    metric = MulticlassClassificationReport(num_classes=5, multidim_average="global")
    report = metric(preds, target)
    assert "0" in report
    assert "4" in report
    assert "macro" in report


def test_binary_incremental():
    """Test that incremental updates give the same result as batch for binary."""
    target = tensor([0, 1, 0, 1, 0, 1])
    preds = tensor([0, 0, 1, 1, 0, 1])

    metric_batch = BinaryClassificationReport()
    report_batch = metric_batch(preds, target)

    metric_incr = BinaryClassificationReport()
    metric_incr.update(preds[:3], target[:3])
    metric_incr.update(preds[3:], target[3:])
    report_incr = metric_incr.compute()

    for key in report_batch:
        for metric_key in ["precision", "recall", "f1_score", "support"]:
            assert torch.allclose(
                report_batch[key][metric_key], report_incr[key][metric_key], atol=1e-8
            ), f"Mismatch at {key}.{metric_key}"
