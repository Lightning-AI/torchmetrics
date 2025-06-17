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
    micro_metrics: Optional[Dict[str, float]] = None,
    show_micro_avg: bool = False,
    is_multilabel: bool = False,
    preds: Optional[Tensor] = None,
    target: Optional[Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Compute macro, micro, weighted, and samples averages for the classification report."""
    total_support = int(sum(metrics["support"] for metrics in class_metrics.values()))
    num_classes = len(class_metrics)

    averages: Dict[str, Dict[str, Union[float, int]]] = {}
    
    # Add micro average if provided and should be shown
    if micro_metrics is not None and show_micro_avg:
        averages["micro avg"] = {
            "precision": micro_metrics["precision"],
            "recall": micro_metrics["recall"],
            "f1-score": micro_metrics["f1-score"],
            "support": total_support,
        }
    
    # Calculate macro and weighted averages
    for avg_name in ["macro avg", "weighted avg"]:
        is_weighted = avg_name == "weighted avg"

        if total_support == 0:
            avg_precision = avg_recall = avg_f1 = 0.0
        else:
            if is_weighted:
                weights = [float(metrics["support"]) / float(total_support) for metrics in class_metrics.values()]
            else:
                weights = [1.0 / float(num_classes) for _ in range(num_classes)]

            # Calculate weighted metrics more efficiently
            metric_names = ["precision", "recall", "f1-score"]
            avg_metrics = {}
            
            for metric_name in metric_names:
                avg_metrics[metric_name] = sum(
                    float(metrics.get(metric_name, 0.0)) * w 
                    for metrics, w in zip(class_metrics.values(), weights)
                )
            
            avg_precision = avg_metrics["precision"]
            avg_recall = avg_metrics["recall"]
            avg_f1 = avg_metrics["f1-score"]

        averages[avg_name] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1-score": avg_f1,
            "support": total_support,
        }
    
    # Add samples average for multilabel classification
    if is_multilabel and preds is not None and target is not None:
        # Convert to binary predictions
        binary_preds = (preds >= threshold).float()
        
        # Calculate per-sample metrics
        n_samples = preds.shape[0]
        sample_precision = torch.zeros(n_samples, dtype=torch.float32)
        sample_recall = torch.zeros(n_samples, dtype=torch.float32)
        sample_f1 = torch.zeros(n_samples, dtype=torch.float32)
        
        for i in range(n_samples):
            true_positives = torch.sum(binary_preds[i] * target[i])
            pred_positives = torch.sum(binary_preds[i])
            actual_positives = torch.sum(target[i])
            
            if pred_positives > 0:
                sample_precision[i] = true_positives / pred_positives
            if actual_positives > 0:
                sample_recall[i] = true_positives / actual_positives
            if pred_positives > 0 and actual_positives > 0:
                sample_f1[i] = 2 * (sample_precision[i] * sample_recall[i]) / (sample_precision[i] + sample_recall[i])
        
        # Average across samples
        avg_precision = torch.mean(sample_precision).item()
        avg_recall = torch.mean(sample_recall).item()
        avg_f1 = torch.mean(sample_f1).item()
        
        averages["samples avg"] = {
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
    micro_metrics: Optional[Dict[str, float]] = None,
    show_micro_avg: bool = False,
    is_multilabel: bool = False,
    preds: Optional[Tensor] = None,
    target_tensor: Optional[Tensor] = None,
    threshold: float = 0.5,
) -> Union[str, Dict[str, Union[float, Dict[str, Union[float, int]]]]]:
    """Format metrics into a classification report."""
    if output_dict:
        result_dict: Dict[str, Union[float, Dict[str, Union[float, int]]]] = {}

        # Add class metrics
        for i, (class_name, metrics) in enumerate(class_metrics.items()):
            display_name = target_names[i] if target_names is not None and i < len(target_names) else str(class_name)
            result_dict[display_name] = {
                "precision": round(float(metrics["precision"]), digits),
                "recall": round(float(metrics["recall"]), digits),
                "f1-score": round(float(metrics["f1-score"]), digits),
                "support": metrics["support"],
            }

        # Add accuracy (only for non-multilabel) and averages
        if not is_multilabel:
            result_dict["accuracy"] = accuracy
            
        result_dict.update(_compute_averages(
            class_metrics, micro_metrics, show_micro_avg, is_multilabel, preds, target_tensor, threshold
        ))

        return result_dict

    # String formatting
    headers = ["precision", "recall", "f1-score", "support"]
    
    # Convert numpy array to list if necessary
    if target_names is not None and hasattr(target_names, "tolist"):
        target_names = target_names.tolist()
    
    # Calculate widths needed for formatting
    name_width = max(len(str(name)) for name in class_metrics)
    if target_names:
        name_width = max(name_width, max(len(str(name)) for name in target_names))
    
    # Add extra width for average methods
    name_width = max(name_width, len("weighted avg"))
    if is_multilabel:
        name_width = max(name_width, len("samples avg"))
    
    # Determine width for each metric column
    width = max(digits + 6, len(headers[0]))
    
    # Format header
    head = " " * name_width + " "
    for h in headers:
        head += "{:>{width}} ".format(h, width=width)
    
    report_lines = [head, ""]
    
    # Format rows for each class
    for i, (class_name, metrics) in enumerate(class_metrics.items()):
        display_name = target_names[i] if target_names and i < len(target_names) else str(class_name)
        # Right-align the class/label name for scikit-learn compatibility
        row = "{:>{name_width}} ".format(display_name, name_width=name_width)
        
        row += "{:>{width}.{digits}f} ".format(
            metrics.get("precision", 0.0), width=width, digits=digits
        )
        row += "{:>{width}.{digits}f} ".format(
            metrics.get("recall", 0.0), width=width, digits=digits
        )
        row += "{:>{width}.{digits}f} ".format(
            metrics.get("f1-score", 0.0), width=width, digits=digits
        )
        row += "{:>{width}} ".format(
            metrics.get("support", 0), width=width
        )
        report_lines.append(row)
    
    # Add a blank line
    report_lines.append("")
    
    # Format accuracy row (only for non-multilabel)
    if not is_multilabel:
        total_support = sum(metrics["support"] for metrics in class_metrics.values())
        acc_row = "{:>{name_width}} ".format("accuracy", name_width=name_width)
        acc_row += "{:>{width}} ".format("", width=width)
        acc_row += "{:>{width}} ".format("", width=width)
        acc_row += "{:>{width}.{digits}f} ".format(accuracy, width=width, digits=digits)
        acc_row += "{:>{width}} ".format(total_support, width=width)
        report_lines.append(acc_row)
    
    # Format averages rows
    averages = _compute_averages(
        class_metrics, micro_metrics, show_micro_avg, is_multilabel, preds, target_tensor, threshold
    )
    for avg_name, avg_metrics in averages.items():
        row = "{:>{name_width}} ".format(avg_name, name_width=name_width)
        
        row += "{:>{width}.{digits}f} ".format(
            avg_metrics["precision"], width=width, digits=digits
        )
        row += "{:>{width}.{digits}f} ".format(
            avg_metrics["recall"], width=width, digits=digits
        )
        row += "{:>{width}.{digits}f} ".format(
            avg_metrics["f1-score"], width=width, digits=digits
        )
        row += "{:>{width}} ".format(
            avg_metrics["support"], width=width
        )
        report_lines.append(row)
    
    return "\n".join(report_lines)


def _compute_binary_metrics(
    preds: Tensor, target: Tensor, threshold: float, ignore_index: Optional[int], validate_args: bool
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for binary classification."""
    class_metrics = {}

    for class_idx in [0, 1]:
        if class_idx == 0:
            # For class 0 (negative class), we need to invert both preds and target
            # But first we need to handle ignore_index properly
            if ignore_index is not None:
                # Create a mask for valid indices
                mask = target != ignore_index
                # Create inverted target only for valid indices, preserving ignore_index
                inv_target = target.clone()
                inv_target[mask] = 1 - target[mask]
                # Invert predictions for all indices
                inv_preds = 1 - preds
            else:
                inv_preds = 1 - preds
                inv_target = 1 - target

            precision_val = binary_precision(inv_preds, inv_target, threshold, ignore_index=ignore_index, validate_args=validate_args).item()
            recall_val = binary_recall(inv_preds, inv_target, threshold, ignore_index=ignore_index, validate_args=validate_args).item()
            f1_val = binary_fbeta_score(
                inv_preds, inv_target, beta=1.0, threshold=threshold, ignore_index=ignore_index, validate_args=validate_args
            ).item()
        else:
            # For class 1 (positive class), use binary metrics directly
            precision_val = binary_precision(preds, target, threshold, ignore_index=ignore_index, validate_args=validate_args).item()
            recall_val = binary_recall(preds, target, threshold, ignore_index=ignore_index, validate_args=validate_args).item()
            f1_val = binary_fbeta_score(
                preds, target, beta=1.0, threshold=threshold, ignore_index=ignore_index, validate_args=validate_args
            ).item()

        # Calculate support, accounting for ignore_index
        if ignore_index is not None:
            mask = target != ignore_index
            support_val = int(((target == class_idx) & mask).sum().item())
        else:
            support_val = int((target == class_idx).sum().item())

        class_metrics[class_idx] = {
            "precision": precision_val,
            "recall": recall_val,
            "f1-score": f1_val,
            "support": support_val,
        }

    return class_metrics


def _compute_multiclass_metrics(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int], validate_args: bool, top_k: int = 1
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for multiclass classification."""
    # Calculate per-class metrics
    precision_vals = multiclass_precision(
        preds, target, num_classes=num_classes, average=None, top_k=top_k, ignore_index=ignore_index, validate_args=validate_args
    )
    recall_vals = multiclass_recall(
        preds, target, num_classes=num_classes, average=None, top_k=top_k, ignore_index=ignore_index, validate_args=validate_args
    )
    f1_vals = multiclass_fbeta_score(
        preds,
        target,
        beta=1.0,
        num_classes=num_classes,
        average=None,
        top_k=top_k,
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
    preds: Tensor, target: Tensor, num_labels: int, threshold: float, ignore_index: Optional[int], validate_args: bool
) -> Dict[int, Dict[str, Union[float, int]]]:
    """Compute metrics for multilabel classification."""
    # Calculate per-label metrics
    precision_vals = multilabel_precision(
        preds, target, num_labels=num_labels, threshold=threshold, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    recall_vals = multilabel_recall(
        preds, target, num_labels=num_labels, threshold=threshold, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    f1_vals = multilabel_fbeta_score(
        preds, target, beta=1.0, num_labels=num_labels, threshold=threshold, average=None, ignore_index=ignore_index, validate_args=validate_args
    )

    # Calculate support for each label, accounting for ignore_index
    if ignore_index is not None:
        # For multilabel, support is the number of positive labels (target=1) excluding ignore_index
        mask = target != ignore_index
        supports = ((target == 1) & mask).sum(dim=0).int()
    else:
        supports = (target == 1).sum(dim=0).int()

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
    labels: Optional[List[int]] = None,
    top_k: int = 1,
) -> Union[str, Dict[str, Union[float, Dict[str, Union[float, int]]]]]:
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
        labels: Optional list of label indices to include in the report (for multiclass tasks)
        top_k: Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits and task is 'multiclass'.

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.
        
    Examples:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification.classification_report import classification_report
        >>> 
        >>> # Binary classification example
        >>> binary_target = tensor([0, 1, 0, 1])
        >>> binary_preds = tensor([0, 1, 1, 1])
        >>> binary_report = classification_report(
        ...     preds=binary_preds,
        ...     target=binary_target,
        ...     task="binary",
        ...     target_names=['Class 0', 'Class 1'],
        ...     digits=2
        ... )
        >>> print(binary_report) # doctest: +NORMALIZE_WHITESPACE
                              precision  recall f1-score support
        <BLANKLINE>
        Class 0                     1.00    0.50     0.67       2
        Class 1                     0.67    1.00     0.80       2
        <BLANKLINE>
        accuracy                                    0.75       4
        macro avg                   0.83    0.75     0.73       4
        weighted avg                0.83    0.75     0.73       4
        >>> 
        >>> # Multiclass classification example
        >>> multiclass_target = tensor([0, 1, 2, 2, 2])
        >>> multiclass_preds = tensor([0, 0, 2, 2, 1])
        >>> multiclass_report = classification_report(
        ...     preds=multiclass_preds,
        ...     target=multiclass_target,
        ...     task="multiclass",
        ...     num_classes=3,
        ...     target_names=["Class 0", "Class 1", "Class 2"],
        ...     digits=2
        ... )
        >>> print(multiclass_report) # doctest: +NORMALIZE_WHITESPACE
                              precision  recall f1-score support
        <BLANKLINE>
        Class 0                    0.50    1.00     0.67       1
        Class 1                    0.00    0.00     0.00       1
        Class 2                    1.00    0.67     0.80       3
        <BLANKLINE>
        accuracy                                    0.60       5
        macro avg                  0.50    0.56     0.49       5
        weighted avg               0.70    0.60     0.61       5
        >>> 
        >>> # Multilabel classification example
        >>> multilabel_target = tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> multilabel_preds = tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        >>> multilabel_report = classification_report(
        ...     preds=multilabel_preds,
        ...     target=multilabel_target,
        ...     task="multilabel",
        ...     num_labels=3,
        ...     target_names=["Label A", "Label B", "Label C"],
        ...     digits=2
        ... )
        >>> print(multilabel_report) # doctest: +NORMALIZE_WHITESPACE
                              precision  recall f1-score support
        <BLANKLINE>
        Label A                    1.00    1.00     1.00       2
        Label B                    1.00    0.50     0.67       2
        Label C                    0.50    1.00     0.67       1
        <BLANKLINE>
        micro avg                  0.80    0.80     0.80       5
        macro avg                  0.83    0.83     0.78       5
        weighted avg               0.90    0.80     0.80       5
        samples avg                0.83    0.83     0.78       5
    """
    # Determine if micro average should be shown in the report based on classification task
    # Following scikit-learn's logic:
    # - Show for multilabel classification (always)
    # - Show for multiclass when using a subset of classes
    # - Don't show for binary classification (micro avg is same as accuracy)
    # - Don't show for full multiclass classification with all classes (micro avg is same as accuracy)
    show_micro_avg = False
    is_multilabel = task == ClassificationTask.MULTILABEL
    
    # Compute task-specific metrics
    if task == ClassificationTask.BINARY:
        class_metrics = _compute_binary_metrics(preds, target, threshold, ignore_index, validate_args)
        accuracy_val = binary_accuracy(preds, target, threshold, ignore_index=ignore_index, validate_args=validate_args).item()
        
        # Calculate micro metrics (same as accuracy for binary classification)
        micro_metrics = {
            "precision": accuracy_val,
            "recall": accuracy_val,
            "f1-score": accuracy_val
        }
        # For binary classification, don't show micro avg (it's same as accuracy)
        show_micro_avg = False

    elif task == ClassificationTask.MULTICLASS:
        if num_classes is None:
            raise ValueError("num_classes must be provided for multiclass classification")

        class_metrics = _compute_multiclass_metrics(preds, target, num_classes, ignore_index, validate_args, top_k)
        
        # Filter metrics by labels if provided
        if labels is not None:
            # Create a new dict with only the specified labels
            filtered_metrics = {
                class_idx: metrics for class_idx, metrics in class_metrics.items() 
                if class_idx in labels
            }
            class_metrics = filtered_metrics
            show_micro_avg = True  # Always show micro avg when specific labels are requested
        else:
            # For multiclass, check if we have a subset of classes with support
            classes_with_support = sum(1 for metrics in class_metrics.values() if metrics["support"] > 0)
            show_micro_avg = classes_with_support < num_classes

        accuracy_val = multiclass_accuracy(
            preds,
            target,
            num_classes=num_classes,
            average="micro",
            top_k=top_k,
            ignore_index=ignore_index,
            validate_args=validate_args,
        ).item()
        
        # Calculate micro-averaged metrics
        micro_precision = multiclass_precision(
            preds, target, num_classes=num_classes, average="micro", 
            top_k=top_k, ignore_index=ignore_index, validate_args=validate_args
        ).item()
        micro_recall = multiclass_recall(
            preds, target, num_classes=num_classes, average="micro", 
            top_k=top_k, ignore_index=ignore_index, validate_args=validate_args
        ).item()
        micro_f1 = multiclass_fbeta_score(
            preds, target, beta=1.0, num_classes=num_classes, average="micro",
            top_k=top_k, ignore_index=ignore_index, validate_args=validate_args
        ).item()
        
        micro_metrics = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1-score": micro_f1
        }

    elif task == ClassificationTask.MULTILABEL:
        if num_labels is None:
            raise ValueError("num_labels must be provided for multilabel classification")

        class_metrics = _compute_multilabel_metrics(preds, target, num_labels, threshold, ignore_index, validate_args)
        accuracy_val = multilabel_accuracy(
            preds, target, num_labels=num_labels, threshold=threshold, average="micro", ignore_index=ignore_index, validate_args=validate_args
        ).item()
        
        # Calculate micro-averaged metrics
        micro_precision = multilabel_precision(
            preds, target, num_labels=num_labels, threshold=threshold, 
            average="micro", ignore_index=ignore_index, validate_args=validate_args
        ).item()
        micro_recall = multilabel_recall(
            preds, target, num_labels=num_labels, threshold=threshold, 
            average="micro", ignore_index=ignore_index, validate_args=validate_args
        ).item()
        micro_f1 = multilabel_fbeta_score(
            preds, target, beta=1.0, num_labels=num_labels, threshold=threshold, 
            average="micro", ignore_index=ignore_index, validate_args=validate_args
        ).item()
        
        micro_metrics = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1-score": micro_f1
        }
        
        # Always show micro avg for multilabel
        show_micro_avg = True

    else:
        raise ValueError(f"Invalid Classification: expected one of (binary, multiclass, multilabel) but got {task}")

    # Apply zero division handling
    _apply_zero_division_handling(class_metrics, zero_division)

    # Filter metrics by labels if provided - this needs to happen after computing all metrics
    # to ensure proper calculation of overall statistics, but before formatting
    if task == ClassificationTask.MULTICLASS and labels is not None:
        # Create a new dict with only the specified labels
        filtered_metrics = {
            class_idx: metrics for class_idx, metrics in class_metrics.items() 
            if class_idx in labels
        }
        class_metrics = filtered_metrics

    # Convert integer keys to strings for compatibility with _format_report
    class_metrics_str = {str(k): v for k, v in class_metrics.items()}
    
    # Apply zero_division to micro metrics
    for key in micro_metrics:
        micro_metrics[key] = _handle_zero_division(micro_metrics[key], zero_division)

    return _format_report(
        class_metrics_str, 
        accuracy_val, 
        target_names, 
        digits, 
        output_dict, 
        micro_metrics, 
        show_micro_avg, 
        is_multilabel,
        preds if is_multilabel else None,
        target if is_multilabel else None,
        threshold
    )


def binary_classification_report(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, float] = 0.0,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[str, Dict[str, Union[float, Dict[str, Union[float, int]]]]]:
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
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification.classification_report import binary_classification_report
        >>> target = tensor([0, 1, 0, 1])
        >>> preds = tensor([0, 1, 1, 1])
        >>> target_names = ['0', '1']
        >>> report = binary_classification_report(
        ...     preds=preds,
        ...     target=target,
        ...     target_names=target_names,
        ...     digits=2
        ... )
        >>> print(report) # doctest: +NORMALIZE_WHITESPACE
                              precision  recall f1-score support
        <BLANKLINE>
        0                          1.00    0.50     0.67       2
        1                          0.67    1.00     0.80       2
        <BLANKLINE>
        accuracy                                    0.75       4
        macro avg                  0.83    0.75     0.73       4
        weighted avg               0.83    0.75     0.73       4
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
        ignore_index=ignore_index,
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
    labels: Optional[List[int]] = None,
    top_k: int = 1,
) -> Union[str, Dict[str, Union[float, Dict[str, Union[float, int]]]]]:
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
        labels: Optional list of label indices to include in the report
        top_k: Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification.classification_report import multiclass_classification_report
        >>> target = tensor([0, 1, 2, 2, 2])
        >>> preds = tensor([0, 0, 2, 2, 1])
        >>> target_names = ["class 0", "class 1", "class 2"]
        >>> report = multiclass_classification_report(
        ...     preds=preds,
        ...     target=target,
        ...     num_classes=3,
        ...     target_names=target_names,
        ...     digits=2
        ... )
        >>> print(report) # doctest: +NORMALIZE_WHITESPACE
                            precision  recall f1-score support
        <BLANKLINE>
        class 0                  0.50    1.00     0.67       1
        class 1                  0.00    0.00     0.00       1
        class 2                  1.00    0.67     0.80       3
        <BLANKLINE>
        accuracy                                  0.60       5
        macro avg                0.50    0.56     0.49       5
        weighted avg             0.70    0.60     0.61       5
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
        labels=labels,
        top_k=top_k,
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
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[str, Dict[str, Union[float, Dict[str, Union[float, int]]]]]:
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
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness

    Returns:
        If output_dict=True, a dictionary with the classification report data.
        Otherwise, a formatted string with the classification report.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification.classification_report import multilabel_classification_report
        >>> target = tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> preds = tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        >>> target_names = ["Label A", "Label B", "Label C"]
        >>> report = multilabel_classification_report(
        ...     preds=preds,
        ...     target=target,
        ...     num_labels=len(target_names),
        ...     target_names=target_names,
        ...     digits=2,
        ... )
        >>> print(report) # doctest: +NORMALIZE_WHITESPACE
                            precision  recall f1-score support
        <BLANKLINE>
        Label A                  1.00    1.00     1.00       2
        Label B                  1.00    0.50     0.67       2
        Label C                  0.50    1.00     0.67       1
        <BLANKLINE>
        micro avg                0.80    0.80     0.80       5
        macro avg                0.83    0.83     0.78       5
        weighted avg             0.90    0.80     0.80       5
        samples avg              0.83    0.83     0.78       5

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
        ignore_index=ignore_index,
        validate_args=validate_args,
    )
