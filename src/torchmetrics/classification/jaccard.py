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
from typing import Any, Optional, Sequence, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from torchmetrics.functional.classification.jaccard import (
    _jaccard_index_reduce,
    _multiclass_jaccard_index_arg_validation,
    _multilabel_jaccard_index_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BinaryJaccardIndex.plot", "MulticlassJaccardIndex.plot", "MultilabelJaccardIndex.plot"]


class BinaryJaccardIndex(BinaryConfusionMatrix):
    r"""Calculate the Jaccard index for binary tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per element.
      Addtionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bji`` (:class:`~torch.Tensor`): A tensor containing the Binary Jaccard Index.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryJaccardIndex
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> metric = BinaryJaccardIndex()
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryJaccardIndex
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> metric = BinaryJaccardIndex()
        >>> metric(preds, target)
        tensor(0.5000)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

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
        """Compute metric."""
        return _jaccard_index_reduce(self.confmat, average="binary")

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryJaccardIndex
            >>> metric = BinaryJaccardIndex()
            >>> metric.update(rand(10), randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryJaccardIndex
            >>> metric = BinaryJaccardIndex()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2,(10,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class MulticlassJaccardIndex(MulticlassConfusionMatrix):
    r"""Calculate the Jaccard index for multiclass tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcji`` (:class:`~torch.Tensor`): A tensor containing the Multi-class Jaccard Index.

    Args:
        num_classes: Integer specifing the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

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
        """Compute metric."""
        return _jaccard_index_reduce(self.confmat, average=self.average, ignore_index=self.ignore_index)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value per class
            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassJaccardIndex
            >>> metric = MulticlassJaccardIndex(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting a multiple values per class
            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassJaccardIndex
            >>> metric = MulticlassJaccardIndex(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class MultilabelJaccardIndex(MultilabelConfusionMatrix):
    r"""Calculate the Jaccard index for multilabel tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int tensor or float tensor of shape ``(N, C, ...)``. If preds is a
      floating point tensor with values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element. Addtionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlji`` (:class:`~torch.Tensor`): A tensor containing the Multi-label Jaccard Index loss.

    Args:
        num_classes: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

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
        """Compute metric."""
        return _jaccard_index_reduce(self.confmat, average=self.average)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelJaccardIndex
            >>> metric = MultilabelJaccardIndex(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelJaccardIndex
            >>> metric = MultilabelJaccardIndex(num_labels=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class JaccardIndex:
    r"""Calculate the Jaccard index for multilabel tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryJaccardIndex`, :mod:`MulticlassJaccardIndex` and :mod:`MultilabelJaccardIndex` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import randint, tensor
        >>> target = randint(0, 2, (10, 25, 25))
        >>> pred = tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard = JaccardIndex(task="multiclass", num_classes=2)
        >>> jaccard(pred, target)
        tensor(0.9660)
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryJaccardIndex(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassJaccardIndex(num_classes, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelJaccardIndex(num_labels, threshold, average, **kwargs)
        return None
