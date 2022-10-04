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
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.hinge import (
    MulticlassMode,
    _binary_confusion_matrix_format,
    _binary_hinge_loss_arg_validation,
    _binary_hinge_loss_tensor_validation,
    _binary_hinge_loss_update,
    _hinge_compute,
    _hinge_loss_compute,
    _hinge_update,
    _multiclass_confusion_matrix_format,
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
    _multiclass_hinge_loss_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn


class BinaryHingeLoss(Metric):
    r"""Computes the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks. It is
    defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import BinaryHingeLoss
        >>> preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> metric = BinaryHingeLoss()
        >>> metric(preds, target)
        tensor(0.6900)
        >>> metric = BinaryHingeLoss(squared=True)
        >>> metric(preds, target)
        tensor(0.6905)
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        squared: bool = False,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_hinge_loss_arg_validation(squared, ignore_index)
        self.validate_args = validate_args
        self.squared = squared
        self.ignore_index = ignore_index

        self.add_state("measures", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _binary_hinge_loss_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(
            preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False
        )
        measures, total = _binary_hinge_loss_update(preds, target, self.squared)
        self.measures += measures
        self.total += total

    def compute(self) -> Tensor:
        return _hinge_loss_compute(self.measures, self.total)


class MulticlassHingeLoss(Metric):
    r"""Computes the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \text{Hinge loss} = \max\left(0, 1 - \hat{y}_y + \max_{i \ne y} (\hat{y}_i)\right)

    Where :math:`y \in {0, ..., \mathrm{C}}` is the target class (where :math:`\mathrm{C}` is the number of classes),
    and :math:`\hat{y} \in \mathbb{R}^\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import MulticlassHingeLoss
        >>> preds = torch.tensor([[0.25, 0.20, 0.55],
        ...                       [0.55, 0.05, 0.40],
        ...                       [0.10, 0.30, 0.60],
        ...                       [0.90, 0.05, 0.05]])
        >>> target = torch.tensor([0, 1, 2, 0])
        >>> metric = MulticlassHingeLoss(num_classes=3)
        >>> metric(preds, target)
        tensor(0.9125)
        >>> metric = MulticlassHingeLoss(num_classes=3, squared=True)
        >>> metric(preds, target)
        tensor(1.1131)
        >>> metric = MulticlassHingeLoss(num_classes=3, multiclass_mode='one-vs-all')
        >>> metric(preds, target)
        tensor([0.8750, 1.1250, 1.1000])
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        squared: bool = False,
        multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.squared = squared
        self.multiclass_mode = multiclass_mode
        self.ignore_index = ignore_index

        self.add_state(
            "measures",
            default=torch.tensor(0.0)
            if self.multiclass_mode == "crammer-singer"
            else torch.zeros(
                num_classes,
            ),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multiclass_hinge_loss_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index, convert_to_labels=False)
        measures, total = _multiclass_hinge_loss_update(preds, target, self.squared, self.multiclass_mode)
        self.measures += measures
        self.total += total

    def compute(self) -> Tensor:
        return _hinge_loss_compute(self.measures, self.total)


class HingeLoss(Metric):
    r"""Hinge Loss.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes the mean `Hinge loss`_, typically used for Support Vector Machines (SVMs).

    In the binary case it is defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    In the multi-class case, when ``multiclass_mode=None`` (default), ``multiclass_mode=MulticlassMode.CRAMMER_SINGER``
    or ``multiclass_mode="crammer-singer"``, this metric will compute the multi-class hinge loss defined by Crammer and
    Singer as:

    .. math::
        \text{Hinge loss} = \max\left(0, 1 - \hat{y}_y + \max_{i \ne y} (\hat{y}_i)\right)

    Where :math:`y \in {0, ..., \mathrm{C}}` is the target class (where :math:`\mathrm{C}` is the number of classes),
    and :math:`\hat{y} \in \mathbb{R}^\mathrm{C}` is the predicted output per class.

    In the multi-class case when ``multiclass_mode=MulticlassMode.ONE_VS_ALL`` or ``multiclass_mode='one-vs-all'``, this
    metric will use a one-vs-all approach to compute the hinge loss, giving a vector of C outputs where each entry pits
    that class against all remaining classes.

    This metric can optionally output the mean of the squared hinge loss by setting ``squared=True``

    Only accepts inputs with preds shape of (N) (binary) or (N, C) (multi-class) and target shape of (N).

    Args:
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss (default).
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Raises:
        ValueError:
            If ``multiclass_mode`` is not: None, ``MulticlassMode.CRAMMER_SINGER``, ``"crammer-singer"``,
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"``.

    Example (binary case):
        >>> import torch
        >>> from torchmetrics import HingeLoss
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> hinge = HingeLoss()
        >>> hinge(preds, target)
        tensor(0.3000)

    Example (default / multiclass case):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss()
        >>> hinge(preds, target)
        tensor(2.9000)

    Example (multiclass example, one vs all mode):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss(multiclass_mode="one-vs-all")
        >>> hinge(preds, target)
        tensor([2.2333, 1.5000, 1.2333])
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    measure: Tensor
    total: Tensor

    def __new__(
        cls,
        squared: bool = False,
        multiclass_mode: Literal["crammer-singer", "one-vs-all"] = None,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryHingeLoss(squared, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                assert multiclass_mode is not None
                return MulticlassHingeLoss(num_classes, squared, multiclass_mode, **kwargs)
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
        squared: bool = False,
        multiclass_mode: Optional[Union[str, MulticlassMode]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("measure", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        if multiclass_mode not in (None, MulticlassMode.CRAMMER_SINGER, MulticlassMode.ONE_VS_ALL):
            raise ValueError(
                "The `multiclass_mode` should be either None / 'crammer-singer' / MulticlassMode.CRAMMER_SINGER"
                "(default) or 'one-vs-all' / MulticlassMode.ONE_VS_ALL,"
                f" got {multiclass_mode}."
            )

        self.squared = squared
        self.multiclass_mode = multiclass_mode

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        measure, total = _hinge_update(preds, target, squared=self.squared, multiclass_mode=self.multiclass_mode)

        self.measure = measure + self.measure
        self.total = total + self.total

    def compute(self) -> Tensor:
        return _hinge_compute(self.measure, self.total)
