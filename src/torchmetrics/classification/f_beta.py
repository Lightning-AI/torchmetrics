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

from torch import Tensor

from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.f_beta import _fbeta_compute
from torchmetrics.utilities.enums import AverageMethod


class FBetaScore(StatScores):
    r"""Computes `F-score`_, specifically:

    .. math::
        F_\beta = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Where :math:`\beta` is some positive real factor. Works with binary, multiclass, and multilabel data.
    Accepts logit scores or probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label logits and probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

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
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The default value (``None``) will be interpreted
            as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"``, ``None``.

    Example:
        >>> import torch
        >>> from torchmetrics import FBetaScore
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f_beta = FBetaScore(num_classes=3, beta=0.5)
        >>> f_beta(preds, target)
        tensor(0.3333)

    """
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.beta = beta
        allowed_average = list(AverageMethod)
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
        """Computes f-beta over state."""
        tp, fp, tn, fn = self._get_final_stats()
        return _fbeta_compute(tp, fp, tn, fn, self.beta, self.ignore_index, self.average, self.mdmc_reduce)


class F1Score(FBetaScore):
    """Computes F1 metric.

    F1 metrics correspond to a harmonic mean of the precision and recall scores.
    Works with binary, multiclass, and multilabel data. Accepts logits or probabilities from a model
    output or integer class values in prediction. Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary ``(0,1)`` predictions, in the case
            of binary or multi-label inputs. Default value of ``0.5`` corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
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
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.
            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Example:
        >>> import torch
        >>> from torchmetrics import F1Score
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1 = F1Score(num_classes=3)
        >>> f1(preds, target)
        tensor(0.3333)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            beta=1.0,
            threshold=threshold,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass,
            **kwargs,
        )
