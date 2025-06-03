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
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
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


class _BaseClassificationReport(Metric):
    """Base class for classification reports with shared functionality."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provided_target_names = target_names
        self.sample_weight = sample_weight
        self.digits = digits
        self.output_dict = output_dict
        self.zero_division = zero_division

        # Add states for tracking data
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric with predictions and targets."""
        self.metrics.update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Union[Dict[str, Any], str]:
        """Compute the classification report."""
        metrics_dict = self.metrics.compute()
        precision, recall, f1, accuracy = self._extract_metrics(metrics_dict)

        target = dim_zero_cat(self.target)
        support = self._compute_support(target)
        preds = dim_zero_cat(self.preds)

        return self._format_report(precision, recall, f1, support, accuracy, preds, target)

    def _extract_metrics(self, metrics_dict: Dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract and format metrics from the metrics dictionary.

        To be implemented by subclasses.

        """
        raise NotImplementedError

    def _compute_support(self, target: Tensor) -> Tensor:
        """Compute support values.

        To be implemented by subclasses.

        """
        raise NotImplementedError

    def _format_report(
        self,
        precision: Tensor,
        recall: Tensor,
        f1: Tensor,
        support: Tensor,
        accuracy: Tensor,
        preds: Tensor,
        target: Tensor,
    ) -> Union[Dict[str, Any], str]:
        """Format the classification report as either a dictionary or string."""
        if self.output_dict:
            return self._format_dict_report(precision, recall, f1, support, accuracy, preds, target)
        return self._format_string_report(precision, recall, f1, support, accuracy)

    def _format_dict_report(
        self,
        precision: Tensor,
        recall: Tensor,
        f1: Tensor,
        support: Tensor,
        accuracy: Tensor,
        preds: Tensor,
        target: Tensor,
    ) -> Dict[str, Any]:
        """Format the classification report as a dictionary."""
        report_dict = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
            "accuracy": accuracy,
            "preds": preds,
            "target": target,
        }

        # Add class-specific entries
        for i, name in enumerate(self.target_names):
            report_dict[name] = {
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1-score": f1[i].item(),
                "support": support[i].item(),
            }

        # Add aggregate metrics
        report_dict["macro avg"] = {
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1-score": f1.mean().item(),
            "support": support.sum().item(),
        }

        # Add weighted average
        weighted_precision = (precision * support).sum() / support.sum()
        weighted_recall = (recall * support).sum() / support.sum()
        weighted_f1 = (f1 * support).sum() / support.sum()

        report_dict["weighted avg"] = {
            "precision": weighted_precision.item(),
            "recall": weighted_recall.item(),
            "f1-score": weighted_f1.item(),
            "support": support.sum().item(),
        }

        return report_dict

    def _format_string_report(
        self,
        precision: Tensor,
        recall: Tensor,
        f1: Tensor,
        support: Tensor,
        accuracy: Tensor,
    ) -> str:
        """Format the classification report as a string."""
        headers = ["precision", "recall", "f1-score", "support"]

        # Set up string formatting
        name_width = max(len(cn) for cn in self.target_names)
        longest_last_line_heading = "weighted avg"
        width = max(name_width, len(longest_last_line_heading))

        # Create the header line with proper spacing
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"

        # Format for rows
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

        # Add result rows
        for i, name in enumerate(self.target_names):
            report += row_fmt.format(
                name,
                precision[i].item(),
                recall[i].item(),
                f1[i].item(),
                int(support[i].item()),
                width=width,
                digits=self.digits,
            )

        # Add blank line
        report += "\n"

        # Add accuracy row - with exact spacing matching sklearn
        report += "{:>{width}s} {:>18} {:>11.{digits}f} {:>9}\n".format(
            "accuracy", "", accuracy.item(), int(support.sum().item()), width=width, digits=self.digits
        )

        # Add macro avg
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()
        report += row_fmt.format(
            "macro avg",
            macro_precision,
            macro_recall,
            macro_f1,
            int(support.sum().item()),
            width=width,
            digits=self.digits,
        )

        # Add weighted avg
        weighted_precision = (precision * support).sum() / support.sum()
        weighted_recall = (recall * support).sum() / support.sum()
        weighted_f1 = (f1 * support).sum() / support.sum()

        report += row_fmt.format(
            "weighted avg",
            weighted_precision.item(),
            weighted_recall.item(),
            weighted_f1.item(),
            int(support.sum().item()),
            width=width,
            digits=self.digits,
        )

        return report

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        """
        if not self.output_dict:
            raise ValueError("Plotting is only supported when output_dict=True")
        return self._plot(val, ax)


class BinaryClassificationReport(_BaseClassificationReport):
    r"""Compute precision, recall, F-measure and support for binary classification tasks.

    The classification report provides detailed metrics for each class in a binary classification task:
    precision, recall, F1-score, and support.

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

        \text{Recall} = \frac{TP}{TP + FN}

        \text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}

        \text{Support} = \sum_i^N 1(y_i = k)

    Where :math:`TP` is true positives, :math:`FP` is false positives, :math:`FN` is false negatives,
    :math:`y` is a tensor of target values, :math:`k` is the class, and :math:`N` is the number of samples.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, ...)`` where ``N`` is
          the batch size. If preds is a floating point tensor with values outside [0,1] range we consider
          the input to be logits and will auto apply sigmoid per element. Additionally, we convert to int
          tensor with thresholding using the value in ``threshold``.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, ...)`` where ``N`` is the batch size.

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        target_names: Optional list of names for each class
        sample_weight: Optional weights for each sample
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero

    Example (with int tensors):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([1, 0, 1, 1, 0, 1])
        >>> metric = ClassificationReport(
        ...     task="binary",
        ...     num_classes=2,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> test_result = metric.compute()
        >>> print(test_result)
                            precision    recall  f1-score   support

                        0        0.50      0.33      0.43         3
                        1        0.50      0.67      0.57         3

                accuracy                             0.50         6
               macro avg         0.50      0.50      0.50         6
            weighted avg         0.50      0.50      0.50         6

    """

    def __init__(
        self,
        threshold: float = 0.5,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            **kwargs,
        )
        self.threshold = threshold
        self.task = "binary"
        self.num_classes = 2

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = ["0", "1"]

        # Initialize metrics
        self.metrics = MetricCollection({
            "precision": BinaryPrecision(threshold=self.threshold),
            "recall": BinaryRecall(threshold=self.threshold),
            "f1": BinaryF1Score(threshold=self.threshold),
            "accuracy": BinaryAccuracy(threshold=self.threshold),
        })

    def _extract_metrics(self, metrics_dict: Dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract and format metrics from the metrics dictionary for binary classification."""
        # For binary classification, we need to create per-class metrics
        precision = torch.tensor([1 - metrics_dict["precision"], metrics_dict["precision"]])
        recall = torch.tensor([1 - metrics_dict["recall"], metrics_dict["recall"]])
        f1 = torch.tensor([1 - metrics_dict["f1"], metrics_dict["f1"]])
        accuracy = metrics_dict["accuracy"]
        return precision, recall, f1, accuracy

    def _compute_support(self, target: Tensor) -> Tensor:
        """Compute support values for binary classification."""
        return torch.bincount(target.int(), minlength=self.num_classes).float()


class MulticlassClassificationReport(_BaseClassificationReport):
    r"""Compute precision, recall, F-measure and support for multiclass classification tasks.

    The classification report provides detailed metrics for each class in a multiclass classification task:
    precision, recall, F1-score, and support.

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

        \text{Recall} = \frac{TP}{TP + FN}

        \text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}

        \text{Support} = \sum_i^N 1(y_i = k)

    Where :math:`TP` is true positives, :math:`FP` is false positives, :math:`FN` is false negatives,
    :math:`y` is a tensor of target values, :math:`k` is the class, and :math:`N` is the number of samples.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions. If preds is a floating point tensor with values
          outside [0,1] range we consider the input to be logits and will auto apply softmax per sample.
          Additionally, we convert to int tensor with argmax.
        - ``target`` (:class:`~torch.Tensor`): A tensor of integer targets.

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        num_classes: Number of classes in the dataset
        target_names: Optional list of names for each class
        sample_weight: Optional weights for each sample
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero

    Example (with int tensors):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([2, 1, 0, 1, 0, 1])
        >>> preds = tensor([2, 0, 1, 1, 0, 1])
        >>> metric = ClassificationReport(
        ...     task="multiclass",
        ...     num_classes=3,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> print(metric.compute())
                          precision    recall  f1-score   support

                       0       0.50      0.50      0.50         2
                       1       0.67      0.67      0.67         3
                       2       1.00      1.00      1.00         1

                accuracy                           0.67         6
               macro avg       0.72      0.72      0.72         6
            weighted avg       0.67      0.67      0.67         6

    """

    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            **kwargs,
        )
        self.task = "multiclass"
        self.num_classes = num_classes

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_classes)]

        # Initialize metrics
        self.metrics = MetricCollection({
            "precision": MulticlassPrecision(num_classes=num_classes, average=None),
            "recall": MulticlassRecall(num_classes=num_classes, average=None),
            "f1": MulticlassF1Score(num_classes=num_classes, average=None),
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        })

    def _extract_metrics(self, metrics_dict: Dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract and format metrics from the metrics dictionary for multiclass classification."""
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]
        f1 = metrics_dict["f1"]
        accuracy = metrics_dict["accuracy"]
        return precision, recall, f1, accuracy

    def _compute_support(self, target: Tensor) -> Tensor:
        """Compute support values for multiclass classification."""
        return torch.bincount(target.int(), minlength=self.num_classes).float()


class MultilabelClassificationReport(_BaseClassificationReport):
    r"""Compute precision, recall, F-measure and support for multilabel classification tasks.

    The classification report provides detailed metrics for each class in a multilabel classification task:
    precision, recall, F1-score, and support.

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

        \text{Recall} = \frac{TP}{TP + FN}

        \text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}

        \text{Support} = \sum_i^N 1(y_i = k)

    Where :math:`TP` is true positives, :math:`FP` is false positives, :math:`FN` is false negatives,
    :math:`y` is a tensor of target values, :math:`k` is the class, and :math:`N` is the number of samples.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, C)`` where ``N`` is the batch size and ``C`` is
          the number of labels. If preds is a floating point tensor with values outside [0,1] range we consider
          the input to be logits and will auto apply sigmoid per element. Additionally, we convert to int
          tensor with thresholding using the value in ``threshold``.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, C)`` where ``N`` is the batch size and ``C`` is
          the number of labels.

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Args:
        num_labels: Number of labels in the dataset
        target_names: Optional list of names for each label
        threshold: Threshold for transforming probability to binary (0,1) predictions
        sample_weight: Optional weights for each sample
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero

    Example (with int tensors):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> labels = ['A', 'B', 'C']
        >>> target = tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        >>> preds = tensor([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = ClassificationReport(
        ...     task="multilabel",
        ...     num_labels=len(labels),
        ...     target_names=labels,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> test_result = metric.compute()
        >>> print(test_result)
                          precision    recall  f1-score   support

                       A       1.00      1.00      1.00         2
                       B       1.00      1.00      1.00         2
                       C       0.50      0.50      0.50         2

                accuracy                           0.78         6
               macro avg       0.83      0.83      0.83         6
            weighted avg       0.83      0.83      0.83         6

    """

    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        target_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
            **kwargs,
        )
        self.threshold = threshold
        self.task = "multilabel"
        self.num_labels = num_labels

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_labels)]

        # Initialize metrics
        self.metrics = MetricCollection({
            "precision": MultilabelPrecision(num_labels=num_labels, average=None, threshold=self.threshold),
            "recall": MultilabelRecall(num_labels=num_labels, average=None, threshold=self.threshold),
            "f1": MultilabelF1Score(num_labels=num_labels, average=None, threshold=self.threshold),
            "accuracy": MultilabelAccuracy(num_labels=num_labels, average="micro", threshold=self.threshold),
        })

    def _extract_metrics(self, metrics_dict: Dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract and format metrics from the metrics dictionary for multilabel classification."""
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]
        f1 = metrics_dict["f1"]
        accuracy = metrics_dict["accuracy"]
        return precision, recall, f1, accuracy

    def _compute_support(self, target: Tensor) -> Tensor:
        """Compute support values for multilabel classification."""
        return torch.sum(target, dim=0)


class ClassificationReport(_ClassificationTaskWrapper):
    r"""Compute precision, recall, F-measure and support for each class.

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

        \text{Recall} = \frac{TP}{TP + FN}

        \text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}

        \text{Support} = \sum_i^N 1(y_i = k)

    Where :math:`TP` is true positives, :math:`FP` is false positives, :math:`FN` is false negatives,
    :math:`y` is a tensor of target values, :math:`k` is the class, and :math:`N` is the number of samples.

    This module is a simple wrapper that computes per-class metrics and produces a formatted report.
    The report shows the main classification metrics for each class and includes micro and macro averages.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets

    As output to ``forward`` and ``compute`` the metric returns either:

        - A formatted string report if ``output_dict=False``
        - A dictionary of metrics if ``output_dict=True``

    Example (Binary Classification):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([1, 0, 1, 1, 0, 1])
        >>> metric = ClassificationReport(
        ...     task="binary",
        ...     num_classes=2,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> test_result = metric.compute()
        >>> print(test_result)
                            precision    recall  f1-score   support

                        0        0.50      0.33      0.43         3
                        1        0.50      0.67      0.57         3

                accuracy                             0.50         6
               macro avg         0.50      0.50      0.50         6
            weighted avg         0.50      0.50      0.50         6

    Example (Multiclass Classification):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([2, 1, 0, 1, 0, 1])
        >>> preds = tensor([2, 0, 1, 1, 0, 1])
        >>> metric = ClassificationReport(
        ...     task="multiclass",
        ...     num_classes=3,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> print(metric.compute())
                          precision    recall  f1-score   support

                       0       0.50      0.50      0.50         2
                       1       0.67      0.67      0.67         3
                       2       1.00      1.00      1.00         1

                accuracy                           0.67         6
               macro avg       0.72      0.72      0.72         6
            weighted avg       0.67      0.67      0.67         6

    Example (Multilabel Classification):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> labels = ['A', 'B', 'C']
        >>> target = tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        >>> preds = tensor([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = ClassificationReport(
        ...     task="multilabel",
        ...     num_labels=len(labels),
        ...     target_names=labels,
        ...     output_dict=False,
        ... )
        >>> metric.update(preds, target)
        >>> test_result = metric.compute()
        >>> print(test_result)
                          precision    recall  f1-score   support

                       A       1.00      1.00      1.00         2
                       B       1.00      1.00      1.00         2
                       C       0.50      0.50      0.50         2

                accuracy                           0.78         6
               macro avg       0.83      0.83      0.83         6
            weighted avg       0.83      0.83      0.83         6

    """

    def __new__(  # type: ignore[misc]
        cls: type["ClassificationReport"],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)

        common_kwargs = {
            "target_names": target_names,
            "sample_weight": sample_weight,
            "digits": digits,
            "output_dict": output_dict,
            "zero_division": zero_division,
            **kwargs,
        }

        if task == ClassificationTask.BINARY:
            return BinaryClassificationReport(threshold=threshold, **common_kwargs)

        if task == ClassificationTask.MULTICLASS:
            return MulticlassClassificationReport(num_classes=num_classes, **common_kwargs)

        if task == ClassificationTask.MULTILABEL:
            return MultilabelClassificationReport(num_labels=num_labels, threshold=threshold, **common_kwargs)

        raise ValueError(f"Not handled value: {task}")
