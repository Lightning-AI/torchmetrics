# Copyright The PyTorch Lightning team.
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
from typing import Any, Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix, MultilabelConfusionMatrix
from torchmetrics.functional.classification.matthews_corrcoef import (
    _matthews_corrcoef_compute,
    _matthews_corrcoef_reduce,
    _matthews_corrcoef_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn


class BinaryMatthewsCorrCoef(BinaryConfusionMatrix):
    r"""Calculates `Matthews correlation coefficient`_ for binary tasks. This metric measures the general
    correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryMatthewsCorrCoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> metric = BinaryMatthewsCorrCoef()
        >>> metric(preds, target)
        tensor(0.5774)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryMatthewsCorrCoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> metric = BinaryMatthewsCorrCoef()
        >>> metric(preds, target)
        tensor(0.5774)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold, ignore_index, normalize=None, validate_args=validate_args, **kwargs)

    def compute(self) -> Tensor:
        return _matthews_corrcoef_reduce(self.confmat)


class MulticlassMatthewsCorrCoef(MulticlassConfusionMatrix):
    r"""Calculates `Matthews correlation coefficient`_ for multiclass tasks. This metric measures the general
    correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.classification import MulticlassMatthewsCorrCoef
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7000)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassMatthewsCorrCoef
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7000)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, ignore_index, normalize=None, validate_args=validate_args, **kwargs)

    def compute(self) -> Tensor:
        return _matthews_corrcoef_reduce(self.confmat)


class MultilabelMatthewsCorrCoef(MultilabelConfusionMatrix):
    r"""Calculates `Matthews correlation coefficient`_ for multilabel tasks. This metric measures the general
    correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MultilabelMatthewsCorrCoef
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelMatthewsCorrCoef
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, threshold, ignore_index, normalize=None, validate_args=validate_args, **kwargs)

    def compute(self) -> Tensor:
        return _matthews_corrcoef_reduce(self.confmat)


class MatthewsCorrCoef(Metric):
    r"""Matthews correlation coefficient.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Calculates `Matthews correlation coefficient`_ that measures the general correlation
    or quality of a classification.

    In the binary case it is defined as:

    .. math::
        MCC = \frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}

    where TP, TN, FP and FN are respectively the true postitives, true negatives,
    false positives and false negatives. Also works in the case of multi-label or
    multi-class input.

    Note:
        This metric produces a multi-dimensional output, so it can not be directly logged.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        threshold: Threshold value for binary or multi-label probabilites.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MatthewsCorrCoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> matthews_corrcoef = MatthewsCorrCoef(num_classes=2)
        >>> matthews_corrcoef(preds, target)
        tensor(0.5774)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    confmat: Tensor

    def __new__(
        cls,
        num_classes: int,
        threshold: float = 0.5,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryMatthewsCorrCoef(threshold, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassMatthewsCorrCoef(num_classes, **kwargs)
            if task == "multilabel":
                assert isinstance(num_labels, int)
                return MultilabelMatthewsCorrCoef(num_labels, threshold, **kwargs)
            raise ValueError(
                f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
            )
        else:
            rank_zero_warn(
                "From v0.10 an `'Binary*'`, `'Multiclass*', `'Multilabel*'` version now exist of each classification"
                " metric. Moving forward we recommend using these versions. This base metric will still work as it did"
                " prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required"
                " and the general order of arguments may change, such that this metric will just function as an single"
                " entrypoint to calling the three specialized versions.",
                DeprecationWarning,
            )
        return super().__new__(cls)

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _matthews_corrcoef_update(preds, target, self.num_classes, self.threshold)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes matthews correlation coefficient."""
        return _matthews_corrcoef_compute(self.confmat)
