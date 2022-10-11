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

from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix
from torchmetrics.functional.classification.cohen_kappa import (
    _binary_cohen_kappa_arg_validation,
    _cohen_kappa_compute,
    _cohen_kappa_reduce,
    _cohen_kappa_update,
    _multiclass_cohen_kappa_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn


class BinaryCohenKappa(BinaryConfusionMatrix):
    r"""Calculates `Cohen's kappa score`_ that measures inter-annotator agreement for binary tasks. It is defined
    as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

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
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryCohenKappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> metric = BinaryCohenKappa()
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryCohenKappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> metric = BinaryCohenKappa()
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
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold, ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _binary_cohen_kappa_arg_validation(threshold, ignore_index, weights)
        self.weights = weights
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        return _cohen_kappa_reduce(self.confmat, self.weights)


class MulticlassCohenKappa(MulticlassConfusionMatrix):
    r"""Calculates `Cohen's kappa score`_ that measures inter-annotator agreement for multiclass tasks. It is
    defined as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

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
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.classification import MulticlassCohenKappa
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassCohenKappa(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6364)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassCohenKappa
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassCohenKappa(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6364)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        self.weights = weights
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        return _cohen_kappa_reduce(self.confmat, self.weights)


class CohenKappa(Metric):
    r"""Cohen Kappa.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Calculates `Cohen's kappa score`_ that measures inter-annotator agreement. It is defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Works with binary, multiclass, and multilabel data.  Accepts probabilities from a model output or
    integer class values in prediction.  Works with multi-dimensional preds and target.

    Forward accepts
        - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes

        - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

        threshold:
            Threshold for transforming probability or logit predictions to binary ``(0,1)`` predictions, in the case
            of binary or multi-label inputs. Default value of ``0.5`` corresponds to input being probabilities.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import CohenKappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> cohenkappa = CohenKappa(num_classes=2)
        >>> cohenkappa(preds, target)
        tensor(0.5000)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    confmat: Tensor

    def __new__(
        cls,
        num_classes: Optional[int] = None,
        weights: Optional[str] = None,
        threshold: float = 0.5,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(weights=weights, ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryCohenKappa(threshold, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassCohenKappa(num_classes, **kwargs)
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
        weights: Optional[str] = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.weights = weights
        self.threshold = threshold

        allowed_weights = ("linear", "quadratic", "none", None)
        if self.weights not in allowed_weights:
            raise ValueError(f"Argument weights needs to one of the following: {allowed_weights}")

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _cohen_kappa_update(preds, target, self.num_classes, self.threshold)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes cohen kappa score."""
        return _cohen_kappa_compute(self.confmat, self.weights)
