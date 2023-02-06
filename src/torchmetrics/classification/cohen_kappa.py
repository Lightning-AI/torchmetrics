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
from typing import Any, Optional

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix
from torchmetrics.functional.classification.cohen_kappa import (
    _binary_cohen_kappa_arg_validation,
    _cohen_kappa_reduce,
    _multiclass_cohen_kappa_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel


class BinaryCohenKappa(BinaryConfusionMatrix):
    r"""Calculate `Cohen's kappa score`_ that measures inter-annotator agreement for binary tasks. It is defined
    as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per element.
      Addtionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bck`` (:class:`~torch.Tensor`): A tensor containing cohen kappa score

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
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryCohenKappa
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> metric = BinaryCohenKappa()
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryCohenKappa
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
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
    r"""Calculate `Cohen's kappa score`_ that measures inter-annotator agreement for multiclass tasks. It is
    defined as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Either an int tensor of shape ``(N, ...)` or float tensor of shape
      ``(N, C, ..)``. If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically
      convert probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcck`` (:class:`~torch.Tensor`): A tensor containing cohen kappa score

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
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassCohenKappa
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassCohenKappa(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6364)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassCohenKappa
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
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


class CohenKappa:
    r"""Calculate `Cohen's kappa score`_ that measures inter-annotator agreement. It is defined as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :mod:`BinaryCohenKappa` and :mod:`MulticlassCohenKappa` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> cohenkappa = CohenKappa(task="multiclass", num_classes=2)
        >>> cohenkappa(preds, target)
        tensor(0.5000)
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({"weights": weights, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCohenKappa(threshold, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            assert isinstance(num_classes, int)
            return MulticlassCohenKappa(num_classes, **kwargs)
