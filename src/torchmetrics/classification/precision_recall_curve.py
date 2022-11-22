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
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BinaryPrecisionRecallCurve(Metric):
    r"""Computes the precision-recall curve for binary tasks. The curve consist of multiple pairs of precision and
    recall values evaluated at different thresholds, such that the tradeoff between the two values can been seen.

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
        (tuple): a tuple of 3 tensors containing:

        - precision: an 1d tensor of size (n_thresholds+1, ) with precision values
        - recall: an 1d tensor of size (n_thresholds+1, ) with recall values
        - thresholds: an 1d tensor of size (n_thresholds, ) with increasing threshold values

    Example:
        >>> from torchmetrics.classification import BinaryPrecisionRecallCurve
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryPrecisionRecallCurve(thresholds=None)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([0.5000, 0.7000, 0.8000]))
        >>> metric = BinaryPrecisionRecallCurve(thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000]),
         tensor([1., 1., 1., 0., 0., 0.]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)

        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat", default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long), dist_reduce_fx="sum"
            )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
        preds, target, _ = _binary_precision_recall_curve_format(preds, target, self.thresholds, self.ignore_index)
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_precision_recall_curve_compute(state, self.thresholds)


class MulticlassPrecisionRecallCurve(Metric):
    r"""Computes the precision-recall curve for multiclass tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

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
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from torchmetrics.classification import MulticlassPrecisionRecallCurve
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=None)
        >>> precision, recall, thresholds = metric(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor(0.7500), tensor(0.7500), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor(0.0500)]
        >>> metric = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.2500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.2500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[1., 1., 1., 1., 0., 0.],
                 [1., 1., 1., 1., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat",
                default=torch.zeros(len(thresholds), num_classes, 2, 2, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multiclass_precision_recall_curve_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target, _ = _multiclass_precision_recall_curve_format(
            preds, target, self.num_classes, self.thresholds, self.ignore_index
        )
        state = _multiclass_precision_recall_curve_update(preds, target, self.num_classes, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multiclass_precision_recall_curve_compute(state, self.num_classes, self.thresholds)


class MultilabelPrecisionRecallCurve(Metric):
    r"""Computes the precision-recall curve for multilabel tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

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
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
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

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between labels). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all labels.

    Example:
        >>> from torchmetrics.classification import MultilabelPrecisionRecallCurve
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> metric = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=None)
        >>> precision, recall, thresholds = metric(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.5000, 0.5000, 1.0000, 1.0000]), tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([0.7500, 1.0000, 1.0000, 1.0000])]
        >>> recall  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([1.0000, 0.6667, 0.3333, 0.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0500, 0.4500, 0.7500]), tensor([0.5500, 0.6500, 0.7500]),
         tensor([0.0500, 0.3500, 0.7500])]
        >>> metric = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat",
                default=torch.zeros(len(thresholds), num_labels, 2, 2, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multilabel_precision_recall_curve_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target, _ = _multilabel_precision_recall_curve_format(
            preds, target, self.num_labels, self.thresholds, self.ignore_index
        )
        state = _multilabel_precision_recall_curve_update(preds, target, self.num_labels, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multilabel_precision_recall_curve_compute(state, self.num_labels, self.thresholds, self.ignore_index)


class PrecisionRecallCurve:
    r"""Computes the precision-recall curve. The curve consist of multiple pairs of precision and recall values
    evaluated at different thresholds, such that the tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryPrecisionRecallCurve`, :mod:`MulticlassPrecisionRecallCurve` and
    :mod:`MultilabelPrecisionRecallCurve` for the specific details of each argument influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pr_curve = PrecisionRecallCurve(task="binary")
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.1000, 0.4000, 0.8000])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor(0.7500), tensor(0.7500), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor(0.0500)]
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        kwargs.update(dict(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return BinaryPrecisionRecallCurve(**kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            return MulticlassPrecisionRecallCurve(num_classes, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelPrecisionRecallCurve(num_labels, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
