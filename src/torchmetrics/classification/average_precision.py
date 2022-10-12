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
from typing import Any, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torchmetrics.functional.classification.average_precision import (
    _average_precision_compute,
    _average_precision_update,
    _binary_average_precision_compute,
    _multiclass_average_precision_arg_validation,
    _multiclass_average_precision_compute,
    _multilabel_average_precision_arg_validation,
    _multilabel_average_precision_compute,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class BinaryAveragePrecision(BinaryPrecisionRecallCurve):
    r"""Computes the average precision (AP) score for binary tasks. The AP score summarizes a precision-recall curve
    as an weighted mean of precisions at each threshold, with the difference in recall from the previous threshold
    as weight:

    .. math::
        AP = \sum{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        A single scalar with the average precision score

    Example:
        >>> from torchmetrics.classification import BinaryAveragePrecision
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryAveragePrecision(thresholds=None)
        >>> metric(preds, target)
        tensor(0.5833)
        >>> metric = BinaryAveragePrecision(thresholds=5)
        >>> metric(preds, target)
        tensor(0.6667)
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_average_precision_compute(state, self.thresholds)


class MulticlassAveragePrecision(MulticlassPrecisionRecallCurve):
    r"""Computes the average precision (AP) score for binary tasks. The AP score summarizes a precision-recall curve
    as an weighted mean of precisions at each threshold, with the difference in recall from the previous threshold
    as weight:

    .. math::
        AP = \sum{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``weighted``: Calculates score for each class and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates score for each class and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with AP score per class.
        If `average="macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from torchmetrics.classification import MulticlassAveragePrecision
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassAveragePrecision(num_classes=5, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.6250)
        >>> metric = MulticlassAveragePrecision(num_classes=5, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500,    nan])
        >>> metric = MulticlassAveragePrecision(num_classes=5, average="macro", thresholds=5)
        >>> metric(preds, target)
        tensor(0.5000)
        >>> metric = MulticlassAveragePrecision(num_classes=5, average=None, thresholds=5)
        >>> metric(preds, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500, -0.0000])
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_average_precision_arg_validation(num_classes, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multiclass_average_precision_compute(state, self.num_classes, self.average, self.thresholds)


class MultilabelAveragePrecision(MultilabelPrecisionRecallCurve):
    r"""Computes the average precision (AP) score for binary tasks. The AP score summarizes a precision-recall curve
    as an weighted mean of precisions at each threshold, with the difference in recall from the previous threshold
    as weight:

    .. math::
        AP = \sum{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifing the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: Calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with AP score per class.
        If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from torchmetrics.classification import MultilabelAveragePrecision
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> metric = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.7500)
        >>> metric = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([0.7500, 0.5833, 0.9167])
        >>> metric = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=5)
        >>> metric(preds, target)
        tensor(0.7778)
        >>> metric = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=5)
        >>> metric(preds, target)
        tensor([0.7500, 0.6667, 0.9167])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multilabel_average_precision_arg_validation(num_labels, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multilabel_average_precision_compute(
            state, self.num_labels, self.average, self.thresholds, self.ignore_index
        )


class AveragePrecision(Metric):
    r"""Average Precision.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes the average precision score, which summarises the precision recall curve into one number. Works for
    both binary and multiclass problems. In the case of multiclass, the values will be calculated based on a one-
    vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            defines the reduction that is applied in the case of multiclass and multilabel input.
            Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes. Cannot be
              used with multiclass input.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support.
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (binary case):
        >>> from torchmetrics import AveragePrecision
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(pos_label=1)
        >>> average_precision(pred, target)
        tensor(1.)

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(num_classes=5, average=None)
        >>> average_precision(pred, target)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __new__(
        cls,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryAveragePrecision(**kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassAveragePrecision(num_classes, average, **kwargs)
            if task == "multilabel":
                assert isinstance(num_labels, int)
                return MultilabelAveragePrecision(num_labels, average, **kwargs)
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
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        allowed_average = ("micro", "macro", "weighted", "none", None)
        if average not in allowed_average:
            raise ValueError(f"Expected argument `average` to be one of {allowed_average}" f" but got {average}")
        self.average = average

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AveragePrecision` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _average_precision_update(
            preds, target, self.num_classes, self.pos_label, self.average
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.

        Returns:
            tensor with average precision. If multiclass return list of such tensors, one for each class
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self.num_classes:
            raise ValueError(f"`num_classes` bas to be positive number, but got {self.num_classes}")
        return _average_precision_compute(preds, target, self.num_classes, self.pos_label, self.average)
