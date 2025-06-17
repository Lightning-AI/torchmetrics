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

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.classification_report import (
    binary_classification_report,
    multiclass_classification_report,
    multilabel_classification_report,
)
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

    # Make mypy aware of the dynamically added states
    preds: List[Tensor]
    target: List[Tensor]

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
        self.target_names: List[str] = []

        # Add states for tracking data
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Compute the classification report using functional interface."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        
        return self._call_functional_report(preds, target)

    def _call_functional_report(self, preds: Tensor, target: Tensor) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Call the appropriate functional classification report."""
        raise NotImplementedError("Subclasses must implement _call_functional_report")

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

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification.classification_report import binary_classification_report
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

    def __init__(
        self,
        threshold: float = 0.5,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        ignore_index: Optional[int] = None,
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
        self.ignore_index = ignore_index

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = ["0", "1"]

    def _call_functional_report(self, preds: Tensor, target: Tensor) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Call binary classification report from functional interface."""
        return binary_classification_report(
            preds=preds,
            target=target,
            threshold=self.threshold,
            target_names=self.target_names,
            digits=self.digits,
            output_dict=self.output_dict,
            zero_division=self.zero_division,
            ignore_index=self.ignore_index,
        )

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
        top_k: Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification.classification_report import multiclass_classification_report
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

    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        target_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Tensor] = None,
        digits: int = 2,
        output_dict: bool = False,
        zero_division: Union[str, int] = "warn",
        ignore_index: Optional[int] = None,
        top_k: int = 1,
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
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.top_k = top_k

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_classes)]

    def _call_functional_report(self, preds: Tensor, target: Tensor) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Call multiclass classification report from functional interface."""
        return multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            target_names=self.target_names,
            digits=self.digits,
            output_dict=self.output_dict,
            zero_division=self.zero_division,
            ignore_index=self.ignore_index,
            top_k=self.top_k,
        )

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

        - ``preds`` (:class:`~torch.Tensor`): A tensor of predictions of shape ``(N, C)`` where ``N`` is the
          batch size and ``C`` is the number of labels. If preds is a floating point tensor with values
          outside [0,1] range we consider the input to be logits and will auto apply sigmoid per element.
          Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
        - ``target`` (:class:`~torch.Tensor`): A tensor of targets of shape ``(N, C)`` where ``N`` is the
          batch size and ``C`` is the number of labels.

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

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification.classification_report import multilabel_classification_report
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
        ignore_index: Optional[int] = None,
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
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        # Set target names if they were provided
        if target_names is not None:
            self.target_names = list(target_names)
        else:
            self.target_names = [str(i) for i in range(num_labels)]

    def _call_functional_report(self, preds: Tensor, target: Tensor) -> Union[Dict[str, Union[float, Dict[str, Union[float, int]]]], str]:
        """Call multilabel classification report from functional interface."""
        return multilabel_classification_report(
            preds=preds,
            target=target,
            num_labels=self.num_labels,
            threshold=self.threshold,
            target_names=self.target_names,
            digits=self.digits,
            output_dict=self.output_dict,
            zero_division=self.zero_division,
            ignore_index=self.ignore_index,
        )

class ClassificationReport(_ClassificationTaskWrapper):
    r"""Compute precision, recall, F-measure and support for each class.

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

        \text{Recall} = \frac{TP}{TP + FN}

        \text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}

        \text{Support} = \sum_i^N 1(y_i = k)

    Where :math:`TP` is true positives, :math:`FP` is false positives, :math:`FN` is false negatives,
    :math:`y` is a tensor of target values, :math:`k` is the class, and :math:`N` is the number of samples.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~torchmetrics.classification.BinaryClassificationReport`, 
    :class:`~torchmetrics.classification.MulticlassClassificationReport` and
    :class:`~torchmetrics.classification.MultilabelClassificationReport` for the specific details of each argument 
    influence and examples.

    Example (Binary Classification):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([0, 1, 0, 1])
        >>> preds = tensor([0, 1, 1, 1])
        >>> target_names = ['0', '1']
        >>> report = ClassificationReport(
        ...     task="binary",
        ...     target_names=target_names,
        ...     digits=2
        ... )
        >>> report.update(preds, target)
        >>> print(report.compute()) # doctest: +NORMALIZE_WHITESPACE
                              precision  recall f1-score support
        <BLANKLINE>
        0                          1.00    0.50     0.67       2
        1                          0.67    1.00     0.80       2
        <BLANKLINE>
        accuracy                                    0.75       4
        macro avg                  0.83    0.75     0.73       4
        weighted avg               0.83    0.75     0.73       4
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
        ignore_index: Optional[int] = None,
        top_k: int = 1,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)

        kwargs.update({
            "target_names": target_names,
            "sample_weight": sample_weight,
            "digits": digits,
            "output_dict": output_dict,
            "zero_division": zero_division,
            "ignore_index": ignore_index,
        })

        if task == ClassificationTask.BINARY:
            return BinaryClassificationReport(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"Optional arg `num_classes` must be type `int` when task is {task}. Got {type(num_classes)}"
                )
            kwargs.update({"top_k": top_k})
            return MulticlassClassificationReport(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"Optional arg `num_labels` must be type `int` when task is {task}. Got {type(num_labels)}"
                )
            return MultilabelClassificationReport(num_labels, **kwargs, threshold=threshold)
        raise ValueError(f"Not handled value: {task}")
