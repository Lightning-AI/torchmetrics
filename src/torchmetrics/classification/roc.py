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

from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torchmetrics.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BinaryROC(BinaryPrecisionRecallCurve):
    r"""Computes the Receiver Operating Characteristic (ROC) for binary tasks. The curve consist of multiple pairs
    of true positive rate (TPR) and false positive rate (FPR) values evaluated at different thresholds, such that
    the tradeoff between the two values can be seen.

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

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

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

        - fpr: an 1d tensor of size (n_thresholds+1, ) with false positive rate values
        - tpr: an 1d tensor of size (n_thresholds+1, ) with true positive rate values
        - thresholds: an 1d tensor of size (n_thresholds, ) with decreasing threshold values

    Example:
        >>> from torchmetrics.classification import BinaryROC
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryROC(thresholds=None)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([1.0000, 0.8000, 0.7000, 0.5000, 0.0000]))
        >>> metric = BinaryROC(thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 1., 1., 1.]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_roc_compute(state, self.thresholds)


class MulticlassROC(MulticlassPrecisionRecallCurve):
    r"""Computes the Receiver Operating Characteristic (ROC) for binary tasks. The curve consist of multiple pairs
    of true positive rate (TPR) and false positive rate (FPR) values evaluated at different thresholds, such that
    the tradeoff between the two values can be seen.

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

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

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

        - fpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with false positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with false positive rate values is returned.
        - tpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with true positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with true positive rate values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with decreasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from torchmetrics.classification import MulticlassROC
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassROC(num_classes=5, thresholds=None)
        >>> fpr, tpr, thresholds = metric(preds, target)
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]),
         tensor([0.0000, 0.3333, 1.0000]), tensor([0., 1.])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0., 0.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.0500])]
        >>> metric = MulticlassROC(num_classes=5, thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 0.]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multiclass_roc_compute(state, self.num_classes, self.thresholds)


class MultilabelROC(MultilabelPrecisionRecallCurve):
    r"""Computes the Receiver Operating Characteristic (ROC) for binary tasks. The curve consist of multiple pairs
    of true positive rate (TPR) and false positive rate (FPR) values evaluated at different thresholds, such that
    the tradeoff between the two values can be seen.

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

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
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
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - fpr: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with false positive rate values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with false positive rate values is returned.
        - tpr: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with true positive rate values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with true positive rate values is returned.
        - thresholds: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds, )
          with decreasing threshold values (length may differ between labels). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all labels.

    Example:
        >>> from torchmetrics.classification import MultilabelROC
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> metric = MultilabelROC(num_labels=3, thresholds=None)
        >>> fpr, tpr, thresholds = metric(preds, target)
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.0000, 0.5000, 1.0000]),
         tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 0., 1.])]
        >>> tpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([0.0000, 0.3333, 0.6667, 1.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.4500, 0.0500]),
         tensor([1.0000, 0.7500, 0.6500, 0.5500, 0.0500]),
         tensor([1.0000, 0.7500, 0.3500, 0.0500])]
        >>> metric = MultilabelROC(num_labels=3, thresholds=5)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000],
                 [0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 1.0000, 1.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.6667, 1.0000]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multilabel_roc_compute(state, self.num_labels, self.thresholds, self.ignore_index)


class ROC:
    r"""Computes the Receiver Operating Characteristic (ROC). The curve consist of multiple pairs of true positive
    rate (TPR) and false positive rate (FPR) values evaluated at different thresholds, such that the tradeoff
    between the two values can be seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryROC`, :mod:`MulticlassROC` and :mod:`MultilabelROC` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> roc = ROC(task="binary")
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([1.0000, 0.9526, 0.8808, 0.7311, 0.5000])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> roc = ROC(task="multiclass", num_classes=4)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500])]

        >>> pred = torch.tensor([[0.8191, 0.3680, 0.1138],
        ...                      [0.3584, 0.7576, 0.1183],
        ...                      [0.2286, 0.3468, 0.1338],
        ...                      [0.8603, 0.0745, 0.1837]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> roc = ROC(task='multilabel', num_labels=3)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr
        [tensor([0., 0., 1., 1., 1.]),
         tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
         tensor([0., 1., 1., 1., 1.])]
        >>> thresholds
        [tensor([1.0000, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.0000, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.0000, 0.1837, 0.1338, 0.1183, 0.1138])]
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
            return BinaryROC(**kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            return MulticlassROC(num_classes, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelROC(num_labels, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
