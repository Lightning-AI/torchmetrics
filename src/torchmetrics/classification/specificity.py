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

from torchmetrics.classification.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
    StatScores,
)
from torchmetrics.functional.classification.specificity import _specificity_compute, _specificity_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import AverageMethod
from torchmetrics.utilities.prints import rank_zero_warn


class BinarySpecificity(BinaryStatScores):
    r"""Computes `Specificity`_ for binary tasks:

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
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
        >>> from torchmetrics.classification import BinarySpecificity
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinarySpecificity()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinarySpecificity
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinarySpecificity()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.classification import BinarySpecificity
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = BinarySpecificity(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.0000, 0.3333])
    """

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average)


class MulticlassSpecificity(MulticlassStatScores):
    r"""Computes `Specificity`_ for multiclass tasks:

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
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
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassSpecificity(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8889)
        >>> metric = MulticlassSpecificity(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.6667, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassSpecificity(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8889)
        >>> metric = MulticlassSpecificity(num_classes=3, average=None)
        >>> metric(preds, target)
        tensor([1.0000, 0.6667, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassSpecificity(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.7500, 0.6556])
        >>> metric = MulticlassSpecificity(num_classes=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[0.7500, 0.7500, 0.7500],
                [0.8000, 0.6667, 0.5000]])
    """

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)


class MultilabelSpecificity(MultilabelStatScores):
    r"""Computes `Specificity`_ for multilabel tasks.

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
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

        multidim_average: Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:
        - If ``multidim_average`` is set to ``global``
          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``
        - If ``multidim_average`` is set to ``samplewise``
          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MultilabelSpecificity
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelSpecificity(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> metric = MultilabelSpecificity(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1., 1., 0.])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelSpecificity
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelSpecificity(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> metric = MultilabelSpecificity(num_labels=3, average=None)
        >>> metric(preds, target)
        tensor([1., 1., 0.])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelSpecificity
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> metric = MultilabelSpecificity(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.0000, 0.3333])
        >>> metric = MultilabelSpecificity(num_labels=3, multidim_average='samplewise', average=None)
        >>> metric(preds, target)
        tensor([[0., 0., 0.],
                [0., 0., 1.]])
    """

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)


class Specificity(StatScores):
    r"""Specificity.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes `Specificity`_:

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
    false positives respecitively. With the use of ``top_k`` parameter, this metric can
    generalize to Specificity@K.

    The reduction method (how the specificity scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tn + fp``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. If this parameter is set for multi-label
            inputs, it will take precedence over ``threshold``. For (multi-dim) multi-class inputs,
            this parameter defaults to 1.

            Should be left unset (``None``) for inputs with label predictions.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.

    Example:
        >>> from torchmetrics import Specificity
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> specificity = Specificity(average='macro', num_classes=3)
        >>> specificity(preds, target)
        tensor(0.6111)
        >>> specificity = Specificity(average='micro')
        >>> specificity(preds, target)
        tensor(0.6250)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __new__(
        cls,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        num_labels: Optional[int] = None,
        multidim_average: Optional[Literal["global", "samplewise"]] = "global",
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            assert multidim_average is not None
            kwargs.update(
                dict(multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args)
            )
            if task == "binary":
                return BinarySpecificity(threshold, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                assert isinstance(top_k, int)
                return MulticlassSpecificity(num_classes, top_k, average, **kwargs)
            if task == "multilabel":
                assert isinstance(num_labels, int)
                return MultilabelSpecificity(num_labels, threshold, average, **kwargs)
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
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        _reduce_options = (AverageMethod.WEIGHTED, AverageMethod.NONE, None)
        if "reduce" not in kwargs:
            kwargs["reduce"] = AverageMethod.MACRO if average in _reduce_options else average
        if "mdmc_reduce" not in kwargs:
            kwargs["mdmc_reduce"] = mdmc_average

        super().__init__(
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.average = average

    def compute(self) -> Tensor:
        """Computes the specificity score based on inputs passed in to ``update`` previously.

        Return:
            The shape of the returned tensor depends on the ``average`` parameter:

            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, tn, fn = self._get_final_stats()
        return _specificity_compute(tp, fp, tn, fn, self.average, self.mdmc_reduce)
