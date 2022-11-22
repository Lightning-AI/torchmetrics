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

from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.metric import Metric


class BinaryPrecision(BinaryStatScores):
    r"""Computes `Precision`_ for binary tasks:

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

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

    Returns:
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryPrecision
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryPrecision()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryPrecision
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryPrecision()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.classification import BinaryPrecision
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = BinaryPrecision(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4000, 0.0000])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "precision", tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )


class MulticlassPrecision(MulticlassStatScores):
    r"""Computes `Precision`_ for multiclass tasks.

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
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

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassPrecision
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassPrecision(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> metric = MulticlassPrecision(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.5000, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassPrecision
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassPrecision(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> metric = MulticlassPrecision(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.5000, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassPrecision
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassPrecision(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.3889, 0.2778])
        >>> metric = MulticlassPrecision(num_classes=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[0.6667, 0.0000, 0.5000],
                [0.0000, 0.5000, 0.3333]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "precision", tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average
        )


class MultilabelPrecision(MultilabelStatScores):
    r"""Computes `Precision`_ for multilabel tasks.

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MultilabelPrecision
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelPrecision(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)
        >>> metric = MultilabelPrecision(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.0000, 0.5000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelPrecision
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelPrecision(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)
        >>> metric = MultilabelPrecision(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.0000, 0.5000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelPrecision
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = MultilabelPrecision(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.3333, 0.0000])
        >>> metric = MultilabelPrecision(num_labels=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.0000]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "precision", tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average
        )


class BinaryRecall(BinaryStatScores):
    r"""Computes `Recall`_ for binary tasks:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

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

    Returns:
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryRecall
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryRecall()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryRecall
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryRecall()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.classification import BinaryRecall
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = BinaryRecall(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.6667, 0.0000])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "recall", tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )


class MulticlassRecall(MulticlassStatScores):
    r"""Computes `Recall`_ for multiclass tasks:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
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

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassRecall
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassRecall(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> metric = MulticlassRecall(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassRecall
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassRecall(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> metric = MulticlassRecall(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassRecall
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassRecall(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5000, 0.2778])
        >>> metric = MulticlassRecall(num_classes=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[1.0000, 0.0000, 0.5000],
                [0.0000, 0.3333, 0.5000]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "recall", tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average
        )


class MultilabelRecall(MultilabelStatScores):
    r"""Computes `Recall`_ for multilabel tasks:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MultilabelRecall
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelRecall(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> metric = MultilabelRecall(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1., 0., 1.])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelRecall
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelRecall(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> metric = MultilabelRecall(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1., 0., 1.])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelRecall
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = MultilabelRecall(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.6667, 0.0000])
        >>> metric = MultilabelRecall(num_labels=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[1., 1., 0.],
                [0., 0., 0.]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "recall", tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average
        )


class Precision:
    r"""Computes `Precision`_:

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryPrecision`, :func:`MulticlassPrecision` and :func:`MultilabelPrecision` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> import torch
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> precision = Precision(task="multiclass", average='macro', num_classes=3)
        >>> precision(preds, target)
        tensor(0.1667)
        >>> precision = Precision(task="multiclass", average='micro', num_classes=3)
        >>> precision(preds, target)
        tensor(0.2500)
    """

    def __new__(
        cls,
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
        assert multidim_average is not None
        kwargs.update(dict(multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return BinaryPrecision(threshold, **kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return MulticlassPrecision(num_classes, top_k, average, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelPrecision(num_labels, threshold, average, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )


class Recall:
    r"""Computes `Recall`_:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryRecall`, :mod:`MulticlassRecall` and :mod:`MultilabelRecall` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> import torch
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> recall = Recall(task="multiclass", average='macro', num_classes=3)
        >>> recall(preds, target)
        tensor(0.3333)
        >>> recall = Recall(task="multiclass", average='micro', num_classes=3)
        >>> recall(preds, target)
        tensor(0.2500)
    """

    def __new__(
        cls,
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
        assert multidim_average is not None
        kwargs.update(dict(multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return BinaryRecall(threshold, **kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return MulticlassRecall(num_classes, top_k, average, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelRecall(num_labels, threshold, average, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
