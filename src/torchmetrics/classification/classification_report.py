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
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.classification_report import (
    _compute_average,
    _compute_per_class_metrics,
)
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


class BinaryClassificationReport(BinaryStatScores):
    r"""Compute a classification report for binary tasks.

    Generates per-class and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors
    and compatible with the torchmetrics stateful metric pattern.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, ...)``.
      If preds is a floating point tensor with values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``report`` (:class:`dict`): A dictionary with keys ``"0"``, ``"1"``, ``"macro"``,
      ``"weighted"``. Each value is a dict with keys ``"precision"``, ``"recall"``,
      ``"f1_score"``, ``"support"``.

    Args:
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryClassificationReport
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryClassificationReport()
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(0.6667)
        >>> report["1"]["recall"]
        tensor(0.6667)
        >>> report["macro"]["f1_score"]
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryClassificationReport
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryClassificationReport()
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(0.6667)
        >>> report["1"]["precision"]
        tensor(0.6667)

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
        )
        self.zero_division = zero_division

    def compute(self) -> Dict[str, Dict[str, Tensor]]:  # type: ignore[override]
        """Compute the classification report."""
        tp, fp, tn, fn = self._final_state()

        # For binary, compute metrics for both classes (0 and 1)
        # Class 0 perspective: positive = label 0 → TP=TN_orig, FP=FN_orig, FN=FP_orig
        tp0, fp0, fn0 = tn, fn, fp  # class 0: positive = label 0
        tp1, fp1, fn1 = tp, fp, fn  # class 1: positive = label 1

        # Stack per-class stats: shape (2,) each
        tp_per_class = torch.stack([tp0, tp1])
        fp_per_class = torch.stack([fp0, fp1])
        fn_per_class = torch.stack([fn0, fn1])
        tn_per_class = torch.stack([tn, tp])

        per_class = _compute_per_class_metrics(
            tp_per_class, fp_per_class, tn_per_class, fn_per_class, self.zero_division
        )
        macro_avg = _compute_average(per_class, "macro", tp_per_class, fp_per_class, fn_per_class)
        weighted_avg = _compute_average(per_class, "weighted", tp_per_class, fp_per_class, fn_per_class)

        def _extract(d: Dict[str, Tensor], idx: int) -> Dict[str, Tensor]:
            return {k: v[idx] if v.ndim > 0 else v for k, v in d.items()}

        return {
            "0": _extract(per_class, 0),
            "1": _extract(per_class, 1),
            "macro": dict(macro_avg),
            "weighted": dict(weighted_avg),
        }

    def plot(self, val: Optional[Dict[str, Dict[str, Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot the classification report as a table.

        Args:
            val: The report dict to plot. If ``None``, calls :meth:`compute`.
            ax: An matplotlib axis object. If ``None``, creates a new figure.

        """
        val = val or self.compute()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4)) if ax is None else (ax.get_figure(), ax)

        row_labels = list(val.keys())
        col_labels = ["precision", "recall", "f1_score", "support"]
        cell_text = []
        for row_key in row_labels:
            row_data = val[row_key]
            cell_text.append([
                f"{row_data[c].item():.4f}" if c != "support" else f"{row_data[c].item():.0f}" for c in col_labels
            ])

        table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.axis("off")
        ax.set_title("Classification Report")
        return fig, ax


class MulticlassClassificationReport(MulticlassStatScores):
    r"""Compute a classification report for multiclass tasks.

    Generates per-class and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors
    and compatible with the torchmetrics stateful metric pattern.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float
      tensor of shape ``(N, C, ..)``. If preds is a floating point we apply
      ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``report`` (:class:`dict`): A dictionary with per-class keys ``"0"``, ``"1"``, ...,
      ``"{C-1}"`` and summary keys ``"micro"``, ``"macro"``, ``"weighted"``.
      Each value is a dict with keys ``"precision"``, ``"recall"``, ``"f1_score"``, ``"support"``.

    Args:
        num_classes: Integer specifying the number of classes
        top_k:
            Number of highest probability or logit score predictions considered to find
            the correct label. Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassClassificationReport
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassClassificationReport(num_classes=3)
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(1.)
        >>> report["1"]["recall"]
        tensor(1.)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassClassificationReport
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassClassificationReport(num_classes=3)
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(1.)

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=None,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
        )
        self.num_classes = num_classes
        self.zero_division = zero_division

    def compute(self) -> Dict[str, Dict[str, Tensor]]:  # type: ignore[override]
        """Compute the classification report."""
        tp, fp, tn, fn = self._final_state()
        per_class = _compute_per_class_metrics(tp, fp, tn, fn, self.zero_division)

        result: Dict[str, Dict[str, Tensor]] = {}

        # Per-class entries
        assert self.num_classes is not None  # guaranteed by __init__
        for c in range(self.num_classes):
            result[str(c)] = {
                "precision": per_class["precision"][c],
                "recall": per_class["recall"][c],
                "f1_score": per_class["f1_score"][c],
                "support": per_class["support"][c],
            }

        # Summary averages
        result["micro"] = _compute_average(per_class, "micro", tp, fp, fn)
        result["macro"] = _compute_average(per_class, "macro", tp, fp, fn)
        result["weighted"] = _compute_average(per_class, "weighted", tp, fp, fn)

        return result

    def plot(self, val: Optional[Dict[str, Dict[str, Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot the classification report as a table.

        Args:
            val: The report dict to plot. If ``None``, calls :meth:`compute`.
            ax: An matplotlib axis object. If ``None``, creates a new figure.

        """
        val = val or self.compute()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(4, len(val) * 0.5))) if ax is None else (ax.get_figure(), ax)

        row_labels = list(val.keys())
        col_labels = ["precision", "recall", "f1_score", "support"]
        cell_text = []
        for row_key in row_labels:
            row_data = val[row_key]
            cell_text.append([
                f"{row_data[c].item():.4f}" if c != "support" else f"{row_data[c].item():.0f}" for c in col_labels
            ])

        table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.axis("off")
        ax.set_title("Classification Report")
        return fig, ax


class MultilabelClassificationReport(MultilabelStatScores):
    r"""Compute a classification report for multilabel tasks.

    Generates per-label and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors
    and compatible with the torchmetrics stateful metric pattern.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, C, ...)``.
      If preds is a floating point tensor with values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``report`` (:class:`dict`): A dictionary with per-label keys ``"label_0"``,
      ``"label_1"``, ..., ``"label_{L-1}"`` and summary keys ``"micro"``, ``"macro"``,
      ``"weighted"``. Each value is a dict with keys ``"precision"``, ``"recall"``,
      ``"f1_score"``, ``"support"``.

    Args:
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelClassificationReport
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelClassificationReport(num_labels=3)
        >>> report = metric(preds, target)
        >>> report["label_0"]["precision"]
        tensor(1.)
        >>> report["label_1"]["recall"]
        tensor(0.)

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            average=None,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
        )
        self.num_labels = num_labels
        self.zero_division = zero_division

    def compute(self) -> Dict[str, Dict[str, Tensor]]:  # type: ignore[override]
        """Compute the classification report."""
        tp, fp, tn, fn = self._final_state()
        per_class = _compute_per_class_metrics(tp, fp, tn, fn, self.zero_division)

        result: Dict[str, Dict[str, Tensor]] = {}

        # Per-label entries
        for lbl_idx in range(self.num_labels):
            result[f"label_{lbl_idx}"] = {
                "precision": per_class["precision"][lbl_idx],
                "recall": per_class["recall"][lbl_idx],
                "f1_score": per_class["f1_score"][lbl_idx],
                "support": per_class["support"][lbl_idx],
            }

        # Summary averages
        result["micro"] = _compute_average(per_class, "micro", tp, fp, fn)
        result["macro"] = _compute_average(per_class, "macro", tp, fp, fn)
        result["weighted"] = _compute_average(per_class, "weighted", tp, fp, fn)

        return result

    def plot(self, val: Optional[Dict[str, Dict[str, Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot the classification report as a table.

        Args:
            val: The report dict to plot. If ``None``, calls :meth:`compute`.
            ax: An matplotlib axis object. If ``None``, creates a new figure.

        """
        val = val or self.compute()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(4, len(val) * 0.5))) if ax is None else (ax.get_figure(), ax)

        row_labels = list(val.keys())
        col_labels = ["precision", "recall", "f1_score", "support"]
        cell_text = []
        for row_key in row_labels:
            row_data = val[row_key]
            cell_text.append([
                f"{row_data[c].item():.4f}" if c != "support" else f"{row_data[c].item():.0f}" for c in col_labels
            ])

        table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.axis("off")
        ax.set_title("Classification Report")
        return fig, ax


class ClassificationReport(_ClassificationTaskWrapper):
    r"""Compute a classification report with precision, recall, F1-score, and support.

    This is a wrapper metric that dispatches to the task-specific implementation
    (``BinaryClassificationReport``, ``MulticlassClassificationReport``, or
    ``MultilabelClassificationReport``) based on the ``task`` argument.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions
    - ``target`` (:class:`~torch.Tensor`): Ground truth labels

    As output to ``forward`` and ``compute`` the metric returns a dictionary of
    per-class/label and average metrics.

    Args:
        task: Type of classification task. Should be one of:

            - ``"binary"``: Binary classification
            - ``"multiclass"``: Multiclass classification
            - ``"multilabel"``: Multilabel classification

        num_classes: Number of classes (required for ``"multiclass"`` task)
        num_labels: Number of labels (required for ``"multilabel"`` task)
        threshold: Threshold for transforming probability to binary predictions
        top_k: Number of highest probability predictions considered (multiclass only)
        multidim_average: How to handle additional dimensions
        ignore_index: Target value that is ignored
        validate_args: Whether to validate input arguments
        zero_division: Value to return when there is a zero division
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (binary):
        >>> from torch import tensor
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = ClassificationReport(task="binary")
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(0.6667)

    Example (multiclass):
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = ClassificationReport(task="multiclass", num_classes=3)
        >>> report = metric(preds, target)
        >>> report["0"]["precision"]
        tensor(1.)

    Example (multilabel):
        >>> from torchmetrics.classification import ClassificationReport
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = ClassificationReport(task="multilabel", num_labels=3)
        >>> report = metric(preds, target)
        >>> report["label_0"]["precision"]
        tensor(1.)

    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        top_k: int = 1,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task = ClassificationTask.from_str(task)
        self._metric: Union[
            BinaryClassificationReport, MulticlassClassificationReport, MultilabelClassificationReport
        ]

        if self.task == ClassificationTask.BINARY:
            self._metric = BinaryClassificationReport(
                threshold=threshold,
                multidim_average=multidim_average,
                ignore_index=ignore_index,
                validate_args=validate_args,
                zero_division=zero_division,
                **kwargs,
            )
        elif self.task == ClassificationTask.MULTICLASS:
            if num_classes is None:
                raise ValueError(f"`num_classes` is required for `{task}` task.")
            self._metric = MulticlassClassificationReport(
                num_classes=num_classes,
                top_k=top_k,
                multidim_average=multidim_average,
                ignore_index=ignore_index,
                validate_args=validate_args,
                zero_division=zero_division,
                **kwargs,
            )
        elif self.task == ClassificationTask.MULTILABEL:
            if num_labels is None:
                raise ValueError(f"`num_labels` is required for `{task}` task.")
            self._metric = MultilabelClassificationReport(
                num_labels=num_labels,
                threshold=threshold,
                multidim_average=multidim_average,
                ignore_index=ignore_index,
                validate_args=validate_args,
                zero_division=zero_division,
                **kwargs,
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state."""
        self._metric.update(preds, target)

    def compute(self) -> Dict[str, Dict[str, Tensor]]:  # type: ignore[override]
        """Compute the classification report."""
        return self._metric.compute()

    def forward(self, preds: Tensor, target: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """Update and compute the classification report."""
        self.update(preds, target)
        return self.compute()

    def reset(self) -> None:
        """Reset metric state."""
        self._metric.reset()

    def plot(self, val: Optional[Dict[str, Dict[str, Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot the classification report."""
        return self._metric.plot(val=val, ax=ax)
