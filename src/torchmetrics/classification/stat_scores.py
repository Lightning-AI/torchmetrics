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
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_compute,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_compute,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask


class _AbstractStatScores(Metric):
    tp: Union[List[Tensor], Tensor]
    fp: Union[List[Tensor], Tensor]
    tn: Union[List[Tensor], Tensor]
    fn: Union[List[Tensor], Tensor]

    # define common functions
    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "global",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("tp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("tn", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fn", default(), dist_reduce_fx=dist_reduce_fx)

    def _update_state(self, tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> None:
        """Update states depending on multidim_average argument."""
        if self.multidim_average == "samplewise":
            self.tp.append(tp)  # type: ignore[union-attr]
            self.fp.append(fp)  # type: ignore[union-attr]
            self.tn.append(tn)  # type: ignore[union-attr]
            self.fn.append(fn)  # type: ignore[union-attr]
        else:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

    def _final_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        tn = dim_zero_cat(self.tn)
        fn = dim_zero_cat(self.fn)
        return tp, fp, tn, fn


class BinaryStatScores(_AbstractStatScores):
    r"""Compute true positives, false positives, true negatives, false negatives and the support for binary tasks.

    Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on the ``multidim_average`` parameter:

      - If ``multidim_average`` is set to ``global``, the shape will be ``(5,)``
      - If ``multidim_average`` is set to ``samplewise``, the shape will be ``(N, 5)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryStatScores()
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryStatScores()
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (multidim tensors):
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryStatScores(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])

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
        **kwargs: Any,
    ) -> None:
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self._create_state(size=1, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
        preds, target = _binary_stat_scores_format(preds, target, self.threshold, self.ignore_index)
        tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _binary_stat_scores_compute(tp, fp, tn, fn, self.multidim_average)


class MulticlassStatScores(_AbstractStatScores):
    r"""Computes true positives, false positives, true negatives, false negatives and the support for multiclass tasks.

    Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on ``average`` and ``multidim_average`` parameters:

      - If ``multidim_average`` is set to ``global``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
        - If ``average=None/'none'``, the shape will be ``(C, 5)``

      - If ``multidim_average`` is set to ``samplewise``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
        - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average='micro')
        >>> metric(preds, target)
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> mcss = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average=None)
        >>> mcss(preds, target)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes, top_k, average, multidim_average, ignore_index, zero_division
            )
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self._create_state(
            size=1 if (average == "micro" and top_k == 1) else (num_classes or 1), multidim_average=multidim_average
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds, target, self.num_classes, self.multidim_average, self.ignore_index
            )
        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        num_classes = self.num_classes if self.num_classes is not None else 1
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds, target, num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index
        )
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _multiclass_stat_scores_compute(tp, fp, tn, fn, self.average, self.multidim_average)


class MultilabelStatScores(_AbstractStatScores):
    r"""Compute true positives, false positives, true negatives, false negatives and the support for multilabel tasks.

    Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, C, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on ``average`` and ``multidim_average`` parameters:

      - If ``multidim_average`` is set to ``global``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
        - If ``average=None/'none'``, the shape will be ``(C, 5)``

      - If ``multidim_average`` is set to ``samplewise``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
        - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelStatScores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelStatScores(num_labels=3, average='micro')
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])
        >>> mlss = MultilabelStatScores(num_labels=3, average=None)
        >>> mlss(preds, target)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelStatScores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelStatScores(num_labels=3, average='micro')
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])
        >>> mlss = MultilabelStatScores(num_labels=3, average=None)
        >>> mlss(preds, target)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelStatScores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelStatScores(num_labels=3, multidim_average='samplewise', average='micro')
        >>> metric(preds, target)
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])
        >>> mlss = MultilabelStatScores(num_labels=3, multidim_average='samplewise', average=None)
        >>> mlss(preds, target)
        tensor([[[1, 1, 0, 0, 1],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1]],
                [[0, 0, 0, 2, 2],
                 [0, 2, 0, 0, 0],
                 [0, 0, 1, 1, 1]]])

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels, threshold, average, multidim_average, ignore_index, zero_division
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self._create_state(size=num_labels, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds, target, self.num_labels, self.multidim_average, self.ignore_index
            )
        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, self.multidim_average)
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _multilabel_stat_scores_compute(tp, fp, tn, fn, self.average, self.multidim_average)


class StatScores(_ClassificationTaskWrapper):
    r"""Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryStatScores`, :class:`~torchmetrics.classification.MulticlassStatScores`
    and :class:`~torchmetrics.classification.MultilabelStatScores` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> preds  = tensor([1, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> stat_scores = StatScores(task="multiclass", num_classes=3, average='micro')
        >>> stat_scores(preds, target)
        tensor([2, 2, 6, 2, 4])
        >>> stat_scores = StatScores(task="multiclass", num_classes=3, average=None)
        >>> stat_scores(preds, target)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])

    """

    def __new__(  # type: ignore[misc]
        cls: type["StatScores"],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Optional[Literal["global", "samplewise"]] = "global",
        top_k: Optional[int] = 1,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        assert multidim_average is not None  # noqa: S101  # needed for mypy
        kwargs.update({
            "multidim_average": multidim_average,
            "ignore_index": ignore_index,
            "validate_args": validate_args,
        })
        if task == ClassificationTask.BINARY:
            return BinaryStatScores(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            if not isinstance(top_k, int):
                raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
            return MulticlassStatScores(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelStatScores(num_labels, threshold, average, **kwargs)
        raise ValueError(f"Task {task} not supported!")
