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
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.accuracy import MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.f_beta import MulticlassFBetaScore, MultilabelFBetaScore
from torchmetrics.classification.precision_recall import (
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.classification.specificity import MulticlassSpecificity, MultilabelSpecificity
from torchmetrics.classification.stat_scores import MulticlassStatScores, MultilabelStatScores
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryClassificationReport.plot",
        "MulticlassClassificationReport.plot",
        "MultilabelClassificationReport.plot",
        "ClassificationReport.plot",
    ]

__all__ = [
    "BinaryClassificationReport",
    "ClassificationReport",
    "MulticlassClassificationReport",
    "MultilabelClassificationReport",
]

# Metric name aliases mapping to canonical names
_METRIC_ALIASES: Dict[str, str] = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1-score",
    "f1-score": "f1-score",
    "f-measure": "f1-score",
    "accuracy": "accuracy",
    "specificity": "specificity",
}

# Default metrics matching sklearn's classification_report
_DEFAULT_METRICS: List[str] = ["precision", "recall", "f1-score"]

# All supported metrics
_SUPPORTED_METRICS: List[str] = ["precision", "recall", "f1-score", "accuracy", "specificity"]


def _normalize_metric_names(metrics: Optional[List[str]]) -> List[str]:
    """Normalize metric names using aliases and validate them."""
    if metrics is None:
        return _DEFAULT_METRICS.copy()

    normalized = []
    for m in metrics:
        m_lower = m.lower()
        if m_lower not in _METRIC_ALIASES:
            raise ValueError(f"Unknown metric '{m}'. Supported metrics: {list(_METRIC_ALIASES.keys())}")
        canonical = _METRIC_ALIASES[m_lower]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


class _BaseClassificationReportCollection(Metric):
    """Base class for classification reports using MetricCollection internally."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        target_names: Optional[Sequence[str]] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, float] = 0.0,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provided_target_names = target_names
        self.digits = digits
        self.output_dict = output_dict
        self.zero_division = zero_division
        self.requested_metrics = _normalize_metric_names(metrics)
        self.target_names: List[str] = []

        # Will be set by subclasses
        self._collection: MetricCollection
        self._stat_scores: Metric

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update all metrics in the collection with predictions and targets."""
        preds_processed, target_processed = self._preprocess(preds, target)
        self._collection.update(preds_processed, target_processed)
        self._stat_scores.update(preds_processed, target_processed)

    def _preprocess(self, preds: Tensor, target: Tensor) -> tuple:
        """Preprocess predictions and targets.

        Override in subclasses if needed.

        """
        return preds, target

    def compute(self) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Compute and assemble the classification report."""
        # Get all metric results
        results = self._collection.compute()
        stat_results = self._stat_scores.compute()

        # Assemble into sklearn-like format
        report_dict = self._assemble_report(results, stat_results)

        if self.output_dict:
            return report_dict
        return self._format_report(report_dict)

    def _assemble_report(
        self, results: Dict[str, Tensor], stat_results: Tensor
    ) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Assemble metric results into sklearn-like report dict."""
        raise NotImplementedError("Subclasses must implement _assemble_report")

    def _format_report(self, report_dict: Dict[str, Any]) -> str:
        """Format the report dictionary as a string table."""
        # Determine which metrics to show in order
        metric_order = [m for m in self.requested_metrics if m != "accuracy"]

        # Calculate column widths
        headers = [*metric_order, "support"]
        max_target_len = max(len(name) for name in self.target_names) if self.target_names else 10
        max_target_len = max(max_target_len, len("weighted avg"))

        # Build header row
        header_fmt = f"{{:>{max_target_len}s}}" + " {:>10s}" * len(headers)
        lines = [header_fmt.format("", *headers), ""]

        # Build per-class rows
        for name in self.target_names:
            if name in report_dict:
                row = report_dict[name]
                values = []
                for m in metric_order:
                    val = row.get(m, 0.0)
                    values.append(f"{val:.{self.digits}f}")
                support = int(row.get("support", 0))
                values.append(f"{support:>10d}")
                row_fmt = f"{{:<{max_target_len}s}}" + " {:>10s}" * len(values)
                lines.append(row_fmt.format(name, *values))

        lines.append("")

        # Add accuracy row if present
        if "accuracy" in report_dict:
            acc_val = report_dict["accuracy"]
            if isinstance(acc_val, dict):
                acc = acc_val.get("accuracy", acc_val.get("f1-score", 0.0))
                support = int(acc_val.get("support", 0))
            else:
                acc = acc_val
                # Get total support from weighted avg
                support = int(report_dict.get("weighted avg", {}).get("support", 0))

            # Accuracy row has blank columns for per-class metrics, then accuracy, then support
            blank_cols = " " * 10 * (len(metric_order) - 1)
            acc_str = f"{acc:.{self.digits}f}"
            lines.append(f"{'accuracy':<{max_target_len}s}{blank_cols} {acc_str:>10s} {support:>10d}")

        # Add average rows
        for avg_name in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
            if avg_name in report_dict:
                row = report_dict[avg_name]
                values = []
                for m in metric_order:
                    val = row.get(m, 0.0)
                    values.append(f"{val:.{self.digits}f}")
                support = int(row.get("support", 0))
                values.append(f"{support:>10d}")
                row_fmt = f"{{:<{max_target_len}s}}" + " {:>10s}" * len(values)
                lines.append(row_fmt.format(avg_name, *values))

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        super().reset()
        self._collection.reset()
        self._stat_scores.reset()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        if not self.output_dict:
            raise ValueError("Plotting is only supported when output_dict=True")
        return self._plot(val, ax)


class MulticlassClassificationReport(_BaseClassificationReportCollection):
    r"""Compute a classification report with precision, recall, F-measure and support for multiclass tasks.

    This metric wraps a configurable set of classification metrics (precision, recall, F1-score, etc.)
    into a single report similar to sklearn's classification_report.

    .. math::
        \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}

    .. math::
        \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}

    .. math::
        \text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}

    .. math::
        \text{Support}_c = \text{TP}_c + \text{FN}_c

    For average metrics:

    .. math::
        \text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c

    .. math::
        \text{Weighted F1} = \sum_{c=1}^{C} \frac{\text{Support}_c}{N} \cdot \text{F1}_c

    Where:
        - :math:`C` is the number of classes
        - :math:`N` is the total number of samples
        - :math:`c` is the class index
        - :math:`\text{TP}_c, \text{FP}_c, \text{FN}_c` are true positives, false positives,
          and false negatives for class :math:`c`

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, ...)`` or ``(N, C, ...)``
          where ``C`` is the number of classes. Can be either probabilities/logits or class indices.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        num_classes: Number of classes in the dataset
        target_names: Optional list of names for each class. If None, classes will be 0, 1, ..., num_classes-1.
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero. Can be 0, 1, or "warn"
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        top_k: Number of highest probability predictions considered for finding the correct label
        metrics: List of metrics to include in the report. Defaults to ["precision", "recall", "f1-score"].
            Supported metrics: "precision", "recall", "f1-score", "accuracy", "specificity".
            You can use aliases like "f1" or "f-measure" for "f1-score".

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassClassificationReport
        >>> target = tensor([0, 1, 2, 2, 2])
        >>> preds = tensor([0, 0, 2, 2, 1])
        >>> report = MulticlassClassificationReport(num_classes=3)
        >>> report.update(preds, target)
        >>> print(report.compute())  # doctest: +NORMALIZE_WHITESPACE
                       precision     recall   f1-score    support
        <BLANKLINE>
        0                   0.50       1.00       0.67          1
        1                   0.00       0.00       0.00          1
        2                   1.00       0.67       0.80          3
        <BLANKLINE>
        accuracy                                  0.60          5
        macro avg           0.50       0.56       0.49          5
        weighted avg        0.70       0.60       0.61          5

    Example (custom metrics):
        >>> report = MulticlassClassificationReport(
        ...     num_classes=3,
        ...     metrics=["precision", "specificity"]
        ... )
        >>> report.update(preds, target)
        >>> result = report.compute()

    """

    def __init__(
        self,
        num_classes: int,
        target_names: Optional[Sequence[str]] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, float] = 0.0,
        ignore_index: Optional[int] = None,
        top_k: int = 1,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_names=target_names,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            metrics=metrics,
            **kwargs,
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.top_k = top_k

        # Set target names
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_classes)]

        # Common kwargs for metric construction
        common_kwargs = {
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "validate_args": False,
        }

        # Build the metric collection based on requested metrics
        metric_dict: Dict[str, Metric] = {}

        if "precision" in self.requested_metrics:
            metric_dict["precision_none"] = MulticlassPrecision(average=None, top_k=top_k, **common_kwargs)
            metric_dict["precision_macro"] = MulticlassPrecision(average="macro", top_k=top_k, **common_kwargs)
            metric_dict["precision_weighted"] = MulticlassPrecision(average="weighted", top_k=top_k, **common_kwargs)

        if "recall" in self.requested_metrics:
            metric_dict["recall_none"] = MulticlassRecall(average=None, top_k=top_k, **common_kwargs)
            metric_dict["recall_macro"] = MulticlassRecall(average="macro", top_k=top_k, **common_kwargs)
            metric_dict["recall_weighted"] = MulticlassRecall(average="weighted", top_k=top_k, **common_kwargs)

        if "f1-score" in self.requested_metrics:
            metric_dict["f1_none"] = MulticlassFBetaScore(beta=1.0, average=None, top_k=top_k, **common_kwargs)
            metric_dict["f1_macro"] = MulticlassFBetaScore(beta=1.0, average="macro", top_k=top_k, **common_kwargs)
            metric_dict["f1_weighted"] = MulticlassFBetaScore(
                beta=1.0, average="weighted", top_k=top_k, **common_kwargs
            )

        # Always include accuracy for the global accuracy row (like sklearn)
        metric_dict["accuracy"] = MulticlassAccuracy(average="micro", top_k=top_k, **common_kwargs)

        if "specificity" in self.requested_metrics:
            metric_dict["specificity_none"] = MulticlassSpecificity(average=None, top_k=top_k, **common_kwargs)
            metric_dict["specificity_macro"] = MulticlassSpecificity(average="macro", top_k=top_k, **common_kwargs)
            metric_dict["specificity_weighted"] = MulticlassSpecificity(
                average="weighted", top_k=top_k, **common_kwargs
            )

        self._collection = MetricCollection(metric_dict, compute_groups=True)
        self._stat_scores = MulticlassStatScores(average=None, **common_kwargs)

    def _assemble_report(
        self, results: Dict[str, Tensor], stat_results: Tensor
    ) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Assemble multiclass metric results into sklearn-like report dict."""
        report: Dict[str, Any] = {}

        # stat_results shape: (num_classes, 5) -> [tp, fp, tn, fn, support]
        supports = (stat_results[:, 0] + stat_results[:, 3]).int()  # tp + fn
        total_support = int(supports.sum().item())

        # Per-class metrics
        for i, name in enumerate(self.target_names):
            class_metrics: Dict[str, Union[float, int]] = {}
            if "precision_none" in results:
                class_metrics["precision"] = float(results["precision_none"][i].item())
            if "recall_none" in results:
                class_metrics["recall"] = float(results["recall_none"][i].item())
            if "f1_none" in results:
                class_metrics["f1-score"] = float(results["f1_none"][i].item())
            if "specificity_none" in results:
                class_metrics["specificity"] = float(results["specificity_none"][i].item())

            class_metrics["support"] = int(supports[i].item())
            report[name] = class_metrics

        # Accuracy (global)
        if "accuracy" in results:
            report["accuracy"] = float(results["accuracy"].item())

        # Macro average
        macro_avg: Dict[str, Union[float, int]] = {}
        if "precision_macro" in results:
            macro_avg["precision"] = float(results["precision_macro"].item())
        if "recall_macro" in results:
            macro_avg["recall"] = float(results["recall_macro"].item())
        if "f1_macro" in results:
            macro_avg["f1-score"] = float(results["f1_macro"].item())
        if "specificity_macro" in results:
            macro_avg["specificity"] = float(results["specificity_macro"].item())
        macro_avg["support"] = total_support
        report["macro avg"] = macro_avg

        # Weighted average
        weighted_avg: Dict[str, Union[float, int]] = {}
        if "precision_weighted" in results:
            weighted_avg["precision"] = float(results["precision_weighted"].item())
        if "recall_weighted" in results:
            weighted_avg["recall"] = float(results["recall_weighted"].item())
        if "f1_weighted" in results:
            weighted_avg["f1-score"] = float(results["f1_weighted"].item())
        if "specificity_weighted" in results:
            weighted_avg["specificity"] = float(results["specificity_weighted"].item())
        weighted_avg["support"] = total_support
        report["weighted avg"] = weighted_avg

        return report


class BinaryClassificationReport(MulticlassClassificationReport):
    r"""Compute a classification report with precision, recall, F-measure and support for binary tasks.

    This metric wraps a configurable set of classification metrics (precision, recall, F1-score, etc.)
    into a single report similar to sklearn's classification_report.

    Internally, binary classification is treated as a 2-class multiclass problem to provide
    per-class metrics for both class 0 and class 1.

    .. math::
        \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    .. math::
        \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    .. math::
        \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

    .. math::
        \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TP}`, :math:`\text{FP}`, :math:`\text{TN}` and :math:`\text{FN}` represent the number of true
    positives, false positives, true negatives and false negatives respectively.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, ...)``. If preds is a
          floating point tensor with values outside [0,1] range, we consider the input to be logits and
          will auto-apply sigmoid. Additionally, we convert to int tensor with thresholding.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        target_names: Optional list of names for each class. Defaults to ["0", "1"].
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero. Can be 0, 1, or "warn"
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        metrics: List of metrics to include in the report. Defaults to ["precision", "recall", "f1-score"].
            Supported metrics: "precision", "recall", "f1-score", "accuracy", "specificity".

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryClassificationReport
        >>> target = tensor([0, 1, 0, 1])
        >>> preds = tensor([0, 1, 1, 1])
        >>> report = BinaryClassificationReport()
        >>> report.update(preds, target)
        >>> print(report.compute())  # doctest: +NORMALIZE_WHITESPACE
                       precision     recall   f1-score    support
        <BLANKLINE>
        0                   1.00       0.50       0.67          2
        1                   0.67       1.00       0.80          2
        <BLANKLINE>
        accuracy                                  0.75          4
        macro avg           0.83       0.75       0.73          4
        weighted avg        0.83       0.75       0.73          4

    """

    def __init__(
        self,
        threshold: float = 0.5,
        target_names: Optional[Sequence[str]] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, float] = 0.0,
        ignore_index: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if target_names is None:
            target_names = ["0", "1"]

        super().__init__(
            num_classes=2,
            target_names=target_names,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            ignore_index=ignore_index,
            top_k=1,
            metrics=metrics,
            **kwargs,
        )
        self.threshold = threshold

    def _preprocess(self, preds: Tensor, target: Tensor) -> tuple:
        """Convert binary probabilities/logits to class predictions."""
        if preds.is_floating_point():
            # Apply sigmoid if values are outside [0, 1]
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            # Threshold to get class predictions
            preds = (preds >= self.threshold).long()
        return preds, target


class MultilabelClassificationReport(_BaseClassificationReportCollection):
    r"""Compute a classification report with precision, recall, F-measure and support for multilabel tasks.

    This metric wraps a configurable set of classification metrics (precision, recall, F1-score, etc.)
    into a single report similar to sklearn's classification_report.

    .. math::
        \text{Precision}_l = \frac{\text{TP}_l}{\text{TP}_l + \text{FP}_l}

    .. math::
        \text{Recall}_l = \frac{\text{TP}_l}{\text{TP}_l + \text{FN}_l}

    .. math::
        \text{F1}_l = 2 \cdot \frac{\text{Precision}_l \cdot \text{Recall}_l}{\text{Precision}_l + \text{Recall}_l}

    For micro-averaged metrics:

    .. math::
        \text{Micro Precision} = \frac{\sum_l \text{TP}_l}{\sum_l (\text{TP}_l + \text{FP}_l)}

    .. math::
        \text{Micro Recall} = \frac{\sum_l \text{TP}_l}{\sum_l (\text{TP}_l + \text{FN}_l)}

    .. math::
        \text{Micro F1} = \frac{2 \cdot P_{micro} \cdot R_{micro}}{P_{micro} + R_{micro}}

    Where:
        - :math:`L` is the number of labels
        - :math:`l` is the label index
        - :math:`\text{TP}_l, \text{FP}_l, \text{FN}_l` are true positives, false positives,
          and false negatives for label :math:`l`

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, L)`` where ``L`` is
          the number of labels. Can be either probabilities/logits or binary predictions.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, L)`` containing 0s and 1s

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        num_labels: Number of labels in the dataset
        target_names: Optional list of names for each label. If None, labels will be 0, 1, ..., num_labels-1.
        threshold: Threshold for transforming probability to binary (0,1) predictions
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero. Can be 0, 1, or "warn"
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        metrics: List of metrics to include in the report. Defaults to ["precision", "recall", "f1-score"].
            Supported metrics: "precision", "recall", "f1-score", "accuracy", "specificity".

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelClassificationReport
        >>> target = tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> preds = tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        >>> report = MultilabelClassificationReport(num_labels=3)
        >>> report.update(preds, target)
        >>> print(report.compute())  # doctest: +NORMALIZE_WHITESPACE
                       precision     recall   f1-score    support
        <BLANKLINE>
        0                   1.00       1.00       1.00          2
        1                   1.00       0.50       0.67          2
        2                   0.50       1.00       0.67          1
        <BLANKLINE>
        micro avg           0.80       0.80       0.80          5
        macro avg           0.83       0.83       0.78          5
        weighted avg        0.90       0.80       0.80          5
        samples avg         0.83       0.83       0.78          5

    """

    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        target_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, float] = 0.0,
        ignore_index: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_names=target_names,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            metrics=metrics,
            **kwargs,
        )
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index

        # Set target names
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_labels)]

        # Common kwargs for metric construction
        common_kwargs = {
            "num_labels": num_labels,
            "threshold": threshold,
            "ignore_index": ignore_index,
            "validate_args": False,
        }

        # Build the metric collection based on requested metrics
        metric_dict: Dict[str, Metric] = {}

        if "precision" in self.requested_metrics:
            metric_dict["precision_none"] = MultilabelPrecision(average=None, **common_kwargs)
            metric_dict["precision_micro"] = MultilabelPrecision(average="micro", **common_kwargs)
            metric_dict["precision_macro"] = MultilabelPrecision(average="macro", **common_kwargs)
            metric_dict["precision_weighted"] = MultilabelPrecision(average="weighted", **common_kwargs)
            metric_dict["precision_samples"] = MultilabelPrecision(average="samples", **common_kwargs)

        if "recall" in self.requested_metrics:
            metric_dict["recall_none"] = MultilabelRecall(average=None, **common_kwargs)
            metric_dict["recall_micro"] = MultilabelRecall(average="micro", **common_kwargs)
            metric_dict["recall_macro"] = MultilabelRecall(average="macro", **common_kwargs)
            metric_dict["recall_weighted"] = MultilabelRecall(average="weighted", **common_kwargs)
            metric_dict["recall_samples"] = MultilabelRecall(average="samples", **common_kwargs)

        if "f1-score" in self.requested_metrics:
            metric_dict["f1_none"] = MultilabelFBetaScore(beta=1.0, average=None, **common_kwargs)
            metric_dict["f1_micro"] = MultilabelFBetaScore(beta=1.0, average="micro", **common_kwargs)
            metric_dict["f1_macro"] = MultilabelFBetaScore(beta=1.0, average="macro", **common_kwargs)
            metric_dict["f1_weighted"] = MultilabelFBetaScore(beta=1.0, average="weighted", **common_kwargs)
            metric_dict["f1_samples"] = MultilabelFBetaScore(beta=1.0, average="samples", **common_kwargs)

        if "accuracy" in self.requested_metrics:
            metric_dict["accuracy_none"] = MultilabelAccuracy(average=None, **common_kwargs)
            metric_dict["accuracy_micro"] = MultilabelAccuracy(average="micro", **common_kwargs)
            metric_dict["accuracy_macro"] = MultilabelAccuracy(average="macro", **common_kwargs)

        if "specificity" in self.requested_metrics:
            metric_dict["specificity_none"] = MultilabelSpecificity(average=None, **common_kwargs)
            metric_dict["specificity_micro"] = MultilabelSpecificity(average="micro", **common_kwargs)
            metric_dict["specificity_macro"] = MultilabelSpecificity(average="macro", **common_kwargs)
            metric_dict["specificity_weighted"] = MultilabelSpecificity(average="weighted", **common_kwargs)

        self._collection = MetricCollection(metric_dict, compute_groups=True)
        self._stat_scores = MultilabelStatScores(average=None, **common_kwargs)

    def _assemble_report(
        self, results: Dict[str, Tensor], stat_results: Tensor
    ) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Assemble multilabel metric results into sklearn-like report dict."""
        report: Dict[str, Any] = {}

        # stat_results shape: (num_labels, 5) -> [tp, fp, tn, fn, support]
        supports = (stat_results[:, 0] + stat_results[:, 3]).int()  # tp + fn
        total_support = int(supports.sum().item())

        # Per-label metrics
        for i, name in enumerate(self.target_names):
            label_metrics: Dict[str, Union[float, int]] = {}

            if "precision_none" in results:
                label_metrics["precision"] = float(results["precision_none"][i].item())
            if "recall_none" in results:
                label_metrics["recall"] = float(results["recall_none"][i].item())
            if "f1_none" in results:
                label_metrics["f1-score"] = float(results["f1_none"][i].item())
            if "specificity_none" in results:
                label_metrics["specificity"] = float(results["specificity_none"][i].item())
            if "accuracy_none" in results:
                label_metrics["accuracy"] = float(results["accuracy_none"][i].item())

            label_metrics["support"] = int(supports[i].item())
            report[name] = label_metrics

        # Micro average
        micro_avg: Dict[str, Union[float, int]] = {}
        if "precision_micro" in results:
            micro_avg["precision"] = float(results["precision_micro"].item())
        if "recall_micro" in results:
            micro_avg["recall"] = float(results["recall_micro"].item())
        if "f1_micro" in results:
            micro_avg["f1-score"] = float(results["f1_micro"].item())
        if "specificity_micro" in results:
            micro_avg["specificity"] = float(results["specificity_micro"].item())
        if "accuracy_micro" in results:
            micro_avg["accuracy"] = float(results["accuracy_micro"].item())
        micro_avg["support"] = total_support
        report["micro avg"] = micro_avg

        # Macro average
        macro_avg: Dict[str, Union[float, int]] = {}
        if "precision_macro" in results:
            macro_avg["precision"] = float(results["precision_macro"].item())
        if "recall_macro" in results:
            macro_avg["recall"] = float(results["recall_macro"].item())
        if "f1_macro" in results:
            macro_avg["f1-score"] = float(results["f1_macro"].item())
        if "specificity_macro" in results:
            macro_avg["specificity"] = float(results["specificity_macro"].item())
        if "accuracy_macro" in results:
            macro_avg["accuracy"] = float(results["accuracy_macro"].item())
        macro_avg["support"] = total_support
        report["macro avg"] = macro_avg

        # Weighted average
        weighted_avg: Dict[str, Union[float, int]] = {}
        if "precision_weighted" in results:
            weighted_avg["precision"] = float(results["precision_weighted"].item())
        if "recall_weighted" in results:
            weighted_avg["recall"] = float(results["recall_weighted"].item())
        if "f1_weighted" in results:
            weighted_avg["f1-score"] = float(results["f1_weighted"].item())
        if "specificity_weighted" in results:
            weighted_avg["specificity"] = float(results["specificity_weighted"].item())
        weighted_avg["support"] = total_support
        report["weighted avg"] = weighted_avg

        # Samples average
        samples_avg: Dict[str, Union[float, int]] = {}
        if "precision_samples" in results:
            samples_avg["precision"] = float(results["precision_samples"].item())
        if "recall_samples" in results:
            samples_avg["recall"] = float(results["recall_samples"].item())
        if "f1_samples" in results:
            samples_avg["f1-score"] = float(results["f1_samples"].item())
        samples_avg["support"] = total_support
        report["samples avg"] = samples_avg

        return report


class ClassificationReport(_ClassificationTaskWrapper):
    r"""Compute a classification report with precision, recall, F-measure and support for each class.

    This is a wrapper that automatically selects the appropriate task-specific metric based on the ``task``
    argument. It uses a collection of existing TorchMetrics classification metrics internally, allowing
    you to customize which metrics are included in the report.

    .. math::
        \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}

    .. math::
        \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}

    .. math::
        \text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}

    Where:
        - :math:`c` is the class/label index
        - :math:`\text{TP}_c, \text{FP}_c, \text{FN}_c` are true positives, false positives,
          and false negatives for class :math:`c`

    Args:
        task: The classification task type. One of ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        threshold: Threshold for transforming probability to binary (0,1) predictions (for binary/multilabel)
        num_classes: Number of classes (required for multiclass)
        num_labels: Number of labels (required for multilabel)
        target_names: Optional list of names for each class/label
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero. Can be 0, 1, or "warn"
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        top_k: Number of highest probability predictions considered (for multiclass)
        metrics: List of metrics to include in the report. Defaults to ["precision", "recall", "f1-score"].
            Supported metrics: "precision", "recall", "f1-score", "accuracy", "specificity".
            You can use aliases like "f1" or "f-measure" for "f1-score".

    Example (Binary Classification):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([0, 1, 0, 1])
        >>> preds = tensor([0, 1, 1, 1])
        >>> report = ClassificationReport(task="binary")
        >>> report.update(preds, target)
        >>> print(report.compute())  # doctest: +NORMALIZE_WHITESPACE
                       precision     recall   f1-score    support
        <BLANKLINE>
        0                   1.00       0.50       0.67          2
        1                   0.67       1.00       0.80          2
        <BLANKLINE>
        accuracy                                  0.75          4
        macro avg           0.83       0.75       0.73          4
        weighted avg        0.83       0.75       0.73          4

    Example (Custom Metrics):
        >>> report = ClassificationReport(
        ...     task="multiclass",
        ...     num_classes=3,
        ...     metrics=["precision", "recall", "specificity"]
        ... )

    """

    def __new__(
        cls: type["ClassificationReport"],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        target_names: Optional[Sequence[str]] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, float] = 0.0,
        ignore_index: Optional[int] = None,
        top_k: int = 1,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task_enum = ClassificationTask.from_str(task)

        common_kwargs = {
            "target_names": target_names,
            "digits": digits,
            "output_dict": output_dict,
            "zero_division": zero_division,
            "ignore_index": ignore_index,
            "metrics": metrics,
        }

        if task_enum == ClassificationTask.BINARY:
            return BinaryClassificationReport(threshold=threshold, **common_kwargs, **kwargs)

        if task_enum == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"Optional arg `num_classes` must be type `int` when task is {task}. Got {type(num_classes)}"
                )
            return MulticlassClassificationReport(num_classes=num_classes, top_k=top_k, **common_kwargs, **kwargs)

        if task_enum == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"Optional arg `num_labels` must be type `int` when task is {task}. Got {type(num_labels)}"
                )
            return MultilabelClassificationReport(num_labels=num_labels, threshold=threshold, **common_kwargs, **kwargs)

        raise ValueError(f"Not handled value: {task}")
