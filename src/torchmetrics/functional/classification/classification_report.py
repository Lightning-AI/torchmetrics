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
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.accuracy import (
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from torchmetrics.functional.classification.f_beta import (
    binary_fbeta_score,
    multiclass_fbeta_score,
    multilabel_fbeta_score,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
)
from torchmetrics.utilities.enums import ClassificationTask


def _handle_zero_division(value: float, zero_division: Union[str, float]) -> float:
    """Handle NaN values based on zero_division parameter."""
    if torch.isnan(torch.tensor(value)):
        if zero_division == "warn":
            return 0.0
        if isinstance(zero_division, (int, float)):
            return float(zero_division)
    return value


def _compute_averages(
    class_metrics: Dict[str, Dict[str, Union[float, int]]],
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Compute macro and weighted averages for the classification report."""
    total_support = sum(metrics["support"] for metrics in class_metrics.values())
    num_classes = len(class_metrics)

    averages = {}
    for avg_name in ["macro avg", "weighted avg"]:
        is_weighted = avg_name == "weighted avg"

        if total_support == 0:
            avg_precision = avg_recall = avg_f1 = 0
        else:
            if is_weighted:
                weights = [metrics["support"] / total_support for metrics in class_metrics.values()]
            else:
                weights = [1 / num_classes for _ in class_metrics]

            avg_precision = sum(
                metrics.get("precision", 0.0) * w for metrics, w in zip(class_metrics.values(), weights)
            )
            avg_recall = sum(metrics.get("recall", 0.0) * w for metrics, w in zip(class_metrics.values(), weights))
            avg_f1 = sum(metrics.get("f1-score", 0.0) * w for metrics, w in zip(class_metrics.values(), weights))

        averages[avg_name] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1-score": avg_f1,
            "support": total_support,
        }

    return averages


def _format_report(
    class_metrics: Dict[str, Dict[str, Union[float, int]]],
    accuracy: float,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
) -> Union[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Format metrics into a classification report.

    Args:
        class_metrics: Dictionary of class metrics, with class names as keys
        accuracy: Overall accuracy
        target_names: Optional list of names for each class
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report

    Returns:
        Formatted report either as string or dictionary

    """
    if output_dict:
        result_dict = {}

        # Add class metrics
        for i, (class_name, metrics) in enumerate(class_metrics.items()):
            display_name = target_names[i] if target_names is not None and i < len(target_names) else str(class_name)
            result_dict[display_name] = {
                "precision": round(metrics["precision"], digits),
                "recall": round(metrics["recall"], digits),
                "f1-score": round(metrics["f1-score"], digits),
                "support": metrics["support"],
            }

        # Add accuracy and averages
        result_dict["accuracy"] = accuracy
        result_dict.update(_compute_averages(class_metrics))

        return result_dict

    # String formatting
    headers = ["precision", "recall", "f1-score", "support"]
    fmt = "%s" + " " * 8 + " ".join(["%s" for _ in range(len(headers) - 1)]) + " %s"
    report_lines = []
    name_width = max(max(len(str(name)) for name in class_metrics), len("weighted avg")) + 4

    # Convert numpy array to list if necessary
    if target_names is not None and hasattr(target_names, "tolist"):
        target_names = target_names.tolist()

    # Header
    header_line = fmt % ("".ljust(name_width), *[header.rjust(digits + 5) for header in headers])
    report_lines.extend([header_line, ""])

    # Class metrics
    for i, (class_name, metrics) in enumerate(class_metrics.items()):
        display_name = target_names[i] if target_names and i < len(target_names) else str(class_name)
        line = fmt % (
            display_name.ljust(name_width),
            f"{metrics.get('precision', 0.0):.{digits}f}".rjust(digits + 5),
            f"{metrics.get('recall', 0.0):.{digits}f}".rjust(digits + 5),
            f"{metrics.get('f1-score', 0.0):.{digits}f}".rjust(digits + 5),
            str(metrics.get("support", 0)).rjust(digits + 5),
        )
        report_lines.append(line)

    # Accuracy line
    total_support = sum(metrics["support"] for metrics in class_metrics.values())
    report_lines.extend([
        "",
        fmt
        % (
            "accuracy".ljust(name_width),
            "",
            "",
            f"{accuracy:.{digits}f}".rjust(digits + 5),
            str(total_support).rjust(digits + 5),
        ),
    ])

    # Average metrics
    averages = _compute_averages(class_metrics)
    for avg_name, avg_metrics in averages.items():
        line = fmt % (
            avg_name.ljust(name_width),
            f"{avg_metrics['precision']:.{digits}f}".rjust(digits + 5),
            f"{avg_metrics['recall']:.{digits}f}".rjust(digits + 5),
            f"{avg_metrics['f1-score']:.{digits}f}".rjust(digits + 5),
            str(avg_metrics["support"]).rjust(digits + 5),
        )
        report_lines.append(line)

    return "\n".join(report_lines)


def _compute_binary_metrics(
    preds: Tensor, target: Tensor, threshold: float, validate_args: bool
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for binary classification."""
    class_metrics = {}

    for class_idx in [0, 1]:
        if class_idx == 0:
            # Invert for class 0 (negative class)
            inv_preds = 1 - preds 
            inv_target = 1 - target

            precision_val = binary_precision(inv_preds, inv_target, threshold, validate_args=validate_args).item()
            recall_val = binary_recall(inv_preds, inv_target, threshold, validate_args=validate_args).item()
            f1_val = binary_fbeta_score(
                inv_preds, inv_target, beta=1.0, threshold=threshold, validate_args=validate_args
            ).item()
        else:
            # For class 1 (positive class), use binary metrics directly
            precision_val = binary_precision(preds, target, threshold, validate_args=validate_args).item()
            recall_val = binary_recall(preds, target, threshold, validate_args=validate_args).item()
            f1_val = binary_fbeta_score(
                preds, target, beta=1.0, threshold=threshold, validate_args=validate_args
            ).item()

        support_val = int((target == class_idx).sum().item())

        class_metrics[class_idx] = {
            "precision": precision_val,
            "recall": recall_val,
            "f1-score": f1_val,
            "support": support_val,
        }

    return class_metrics


def _compute_multiclass_metrics(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int], validate_args: bool
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for multiclass classification."""
    # Calculate per-class metrics
    precision_vals = multiclass_precision(
        preds, target, num_classes=num_classes, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    recall_vals = multiclass_recall(
        preds, target, num_classes=num_classes, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    f1_vals = multiclass_fbeta_score(
        preds,
        target,
        beta=1.0,
        num_classes=num_classes,
        average=None,
        ignore_index=ignore_index,
        validate_args=validate_args,
    )

    # Calculate support for each class
    if ignore_index is not None:
        mask = target != ignore_index
        class_counts = torch.bincount(target[mask].flatten(), minlength=num_classes)
    else:
        class_counts = torch.bincount(target.flatten(), minlength=num_classes)

    class_metrics = {}
    for class_idx in range(num_classes):
        class_metrics[class_idx] = {
            "precision": precision_vals[class_idx].item(),
            "recall": recall_vals[class_idx].item(),
            "f1-score": f1_vals[class_idx].item(),
            "support": int(class_counts[class_idx].item()),
        }

    return class_metrics


def _compute_multilabel_metrics(
    preds: Tensor, target: Tensor, num_labels: int, threshold: float, validate_args: bool
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for multilabel classification."""
    # Calculate per-label metrics
    precision_vals = multilabel_precision(
        preds, target, num_labels=num_labels, threshold=threshold, average=None, validate_args=validate_args
    )
    recall_vals = multilabel_recall(
        preds, target, num_labels=num_labels, threshold=threshold, average=None, validate_args=validate_args
    )
    f1_vals = multilabel_fbeta_score(
        preds, target, beta=1.0, num_labels=num_labels, threshold=threshold, average=None, validate_args=validate_args
    )

    # Calculate support for each label
    supports = target.sum(dim=0).int()

    class_metrics = {}
    for label_idx in range(num_labels):
        class_metrics[label_idx] = {
            "precision": precision_vals[label_idx].item(),
            "recall": recall_vals[label_idx].item(),
            "f1-score": f1_vals[label_idx].item(),
            "support": int(supports[label_idx].item()),
        }

    return class_metrics


def _apply_zero_division_handling(
    class_metrics: Dict[int, Dict[str, Union[float, int]]], zero_division: Union[str, float]
) -> None:
    """Apply zero division handling to all class metrics in-place."""
    for metrics in class_metrics.values():
        metrics["precision"] = _handle_zero_division(metrics["precision"], zero_division)
        metrics["recall"] = _handle_zero_division(metrics["recall"], zero_division)
        metrics["f1-score"] = _handle_zero_division(metrics["f1-score"], zero_division)


def classification_report(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, float] = 0.0,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Compute a classification report for various classification tasks.

    The classification report shows the precision, recall, F1 score, and support for each class/label.

    Args:
        preds: Tensor with predictions
        target: Tensor with ground truth labels
        task: The classification task - either 'binary', 'multiclass', or 'multilabel'
        threshold: Threshold for converting probabilities to binary predictions (for binary and multilabel tasks)
        num_classes: Number of classes (for multiclass tasks)
        num_labels: Number of labels (for multilabel tasks)
        target_names: Optional list of names for the classes/labels
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero
        ignore_index: Optional index to ignore in the target (for multiclass tasks)
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

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
    # Compute task-specific metrics
    if task == ClassificationTask.BINARY:
        class_metrics = _compute_binary_metrics(preds, target, threshold, validate_args)
        accuracy_val = binary_accuracy(preds, target, threshold, validate_args=validate_args).item()

    elif task == ClassificationTask.MULTICLASS:
        if num_classes is None:
            raise ValueError("num_classes must be provided for multiclass classification")

        class_metrics = _compute_multiclass_metrics(preds, target, num_classes, ignore_index, validate_args)
        accuracy_val = multiclass_accuracy(
            preds,
            target,
            num_classes=num_classes,
            average="micro",
            ignore_index=ignore_index,
            validate_args=validate_args,
        ).item()

    elif task == ClassificationTask.MULTILABEL:
        if num_labels is None:
            raise ValueError("num_labels must be provided for multilabel classification")

        class_metrics = _compute_multilabel_metrics(preds, target, num_labels, threshold, validate_args)
        accuracy_val = multilabel_accuracy(
            preds, target, num_labels=num_labels, threshold=threshold, average="micro", validate_args=validate_args
        ).item()

    else:
        raise ValueError(f"Invalid Classification: expected one of (binary, multiclass, multilabel) but got {task}")

    # Apply zero division handling
    _apply_zero_division_handling(class_metrics, zero_division)

    return _format_report(class_metrics, accuracy_val, target_names, digits, output_dict)


def binary_classification_report(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, float] = 0.0,
    validate_args: bool = True,
) -> Union[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Compute a classification report for binary classification tasks.

    The classification report shows the precision, recall, F1 score, and support for each class.

    Args:
        preds: Tensor with predictions
        target: Tensor with ground truth labels
        threshold: Threshold for converting probabilities to binary predictions
        target_names: Optional list of names for the classes
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

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
    return classification_report(
        preds,
        target,
        task="binary",
        threshold=threshold,
        target_names=target_names,
        digits=digits,
        output_dict=output_dict,
        zero_division=zero_division,
        validate_args=validate_args,
    )


def multiclass_classification_report(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, float] = 0.0,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Compute a classification report for multiclass classification tasks.

    The classification report shows the precision, recall, F1 score, and support for each class.

    Args:
        preds: Tensor with predictions of shape (N, ...) or (N, C, ...) where C is the number of classes
        target: Tensor with ground truth labels of shape (N, ...)
        num_classes: Number of classes
        target_names: Optional list of names for the classes
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero
        ignore_index: Optional index to ignore in the target
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

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
    return classification_report(
        preds,
        target,
        task="multiclass",
        num_classes=num_classes,
        target_names=target_names,
        digits=digits,
        output_dict=output_dict,
        zero_division=zero_division,
        ignore_index=ignore_index,
        validate_args=validate_args,
    )


def multilabel_classification_report(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, float] = 0.0,
    validate_args: bool = True,
) -> Union[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Compute a classification report for multilabel classification tasks.

    The classification report shows the precision, recall, F1 score, and support for each label.

    Args:
        preds: Tensor with predictions of shape (N, L, ...) where L is the number of labels
        target: Tensor with ground truth labels of shape (N, L, ...)
        num_labels: Number of labels
        threshold: Threshold for converting probabilities to binary predictions
        target_names: Optional list of names for the labels
        digits: Number of decimal places to display in the report
        output_dict: If True, return a dict instead of a string report
        zero_division: Value to use when dividing by zero
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

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
    return classification_report(
        preds,
        target,
        task="multilabel",
        num_labels=num_labels,
        threshold=threshold,
        target_names=target_names,
        digits=digits,
        output_dict=output_dict,
        zero_division=zero_division,
        validate_args=validate_args,
    )
