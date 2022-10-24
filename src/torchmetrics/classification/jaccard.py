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
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.functional.classification.jaccard import (
    _jaccard_from_confmat,
    _jaccard_index_reduce,
    _multiclass_jaccard_index_arg_validation,
    _multilabel_jaccard_index_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn


class BinaryJaccardIndex(BinaryConfusionMatrix):
    r"""Calculates the Jaccard index for binary tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

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
        >>> from torchmetrics.classification import BinaryJaccardIndex
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> metric = BinaryJaccardIndex()
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryJaccardIndex
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> metric = BinaryJaccardIndex()
        >>> metric(preds, target)
        tensor(0.5000)
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
        super().__init__(
            threshold=threshold, ignore_index=ignore_index, normalize=None, validate_args=validate_args, **kwargs
        )

    def compute(self) -> Tensor:
        return _jaccard_index_reduce(self.confmat, average="binary")


class MulticlassJaccardIndex(MulticlassConfusionMatrix):
    r"""Calculates the Jaccard index for multiclass tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

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
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, ignore_index=ignore_index, normalize=None, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        self.validate_args = validate_args
        self.average = average

    def compute(self) -> Tensor:
        return _jaccard_index_reduce(self.confmat, average=self.average)


class MultilabelJaccardIndex(MultilabelConfusionMatrix):
    r"""Calculates the Jaccard index for multilabel tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

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
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
            normalize=None,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index, average)
        self.validate_args = validate_args
        self.average = average

    def compute(self) -> Tensor:
        return _jaccard_index_reduce(self.confmat, average=self.average)


class JaccardIndex(ConfusionMatrix):
    r"""Jaccard Index.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes Intersection over union, or `Jaccard index`_:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Where: :math:`A` and :math:`B` are both tensors of the same size, containing integer class values.
    They may be subject to conversion from input data (see description below). Note that it is different from box IoU.

    Works with binary, multiclass and multi-label data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class. Note that if a given class doesn't occur in the
              `preds` or `target`, the value for the class will be ``nan``.

        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1]. By default, no index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of the class index were present in
            ``preds`` AND no instances of the class index were present in ``target``. For example, if we have 3 classes,
            [0, 0] for ``preds``, and [0, 2] for ``target``, then class 1 would be assigned the `absent_score`.
        threshold: Threshold value for binary or multi-label probabilities.
        multilabel: determines if data is multilabel or not.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import JaccardIndex
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard = JaccardIndex(num_classes=2)
        >>> jaccard(pred, target)
        tensor(0.9660)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __new__(
        cls,
        num_classes: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        multilabel: bool = False,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        num_labels: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryJaccardIndex(threshold, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassJaccardIndex(num_classes, average, **kwargs)
            if task == "multilabel":
                assert isinstance(num_labels, int)
                return MultilabelJaccardIndex(num_labels, threshold, average, **kwargs)
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
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        multilabel: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs["normalize"] = kwargs.get("normalize")

        super().__init__(
            num_classes=num_classes,
            threshold=threshold,
            multilabel=multilabel,
            **kwargs,
        )
        self.average = average
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self) -> Tensor:
        """Computes intersection over union (IoU)"""

        if self.multilabel:
            return torch.stack(
                [
                    _jaccard_from_confmat(
                        confmat,
                        2,
                        self.average,
                        self.ignore_index,
                        self.absent_score,
                    )[1]
                    for confmat in self.confmat
                ]
            )
        else:
            return _jaccard_from_confmat(
                self.confmat,
                self.num_classes,
                self.average,
                self.ignore_index,
                self.absent_score,
            )
