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
from torchmetrics.functional.classification.generalized_dice import _generalized_dice_compute


class GeneralizedDiceScore(StatScores):
    r"""Computes the Generalized Dice Score (GDS) metric:

    .. math::
        \text{GDS}=\sum_{i=1}^{C}\frac{2\cdot\text{TP}_i}{(2\cdot\text{TP}_i+\text{FP}_i+\text{FN}_i)\cdot w_i}

    Where :math:`\text{C}` is the number of classes and :math:`\text{TP}_i`, :math:`\text{FP}_i` and :math:`\text{FN}`_i
    represent the numbers of true positives, false positives and false negatives for class :math:`i`, respectively.
    :math:`w_i` represents the weight of class :math:`i`.

    The reduction method (how the generalized dice scores are aggregated) is controlled by the
    ``average`` parameter. Accepts all inputs listed in :ref:`pages/classification:input types`.
    Does not accept multidimensional multi-label data.

    Args:
        num_classes:
            Number of classes.

        weight_type: Defines the type of weighting to apply. Should be one of the following:

            - ``'square'`` [default]: Weight each class by the squared inverse of its support,
              i.e., the inverse of its squared volume - :math:`\frac{1}{(tp + fn)^2}`.
            - ``'simple'``: Weight each class by the inverse of its support, i.e.,
              the inverse of its volume - :math:`\frac{1}{tp + fn}`.

        zero_division:
            The value to use for the score if denominator equals zero. If set to None, the score will be 1 if the
            numerator is also 0, and 0 otherwise.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'samples'`` [default]: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).
            - ``'none'`` or ``None``: Calculate the metric for each sample separately, and return
              the metric for every sample.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label.
            The default value (``None``) will be interpreted as 1.

        multiclass:
            Determines whether the input is multiclass (if True) or multilabel (if False).

        multidim:
            Determines whether the input is multidim or not.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``weight_type`` is not ``"simple"``, ``"square"`` or ``None``.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``num_classes`` is not larger than ``0``.
        ValueError:
            If ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.

    Example:
        >>> import torch
        >>> from torchmetrics import GeneralizedDiceScore
        >>> preds = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> generalized_dice_score = GeneralizedDiceScore(num_classes=3)
        >>> generalized_dice_score(preds, target)
        tensor(0.3478)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        weight_type: str = "square",
        zero_division: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "samples",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: bool = True,
        multidim: bool = True,
        **kwargs: Any,
    ) -> None:
        allowed_weight_type = ("square", "simple", None)
        if weight_type not in allowed_weight_type:
            raise ValueError(f"The `weight_type` has to be one of {allowed_weight_type}, got {weight_type}.")

        allowed_average = ("samples", "none", None)
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        if ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
            raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

        # Provide "mdmc_reduce" and "reduce" as kwargs
        kwargs["mdmc_reduce"] = "samplewise"
        kwargs["reduce"] = "macro" if multidim else None

        super().__init__(
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.multidim = multidim
        self.average = average
        self.weight_type = weight_type
        self.zero_division = zero_division

    def compute(self) -> Tensor:
        """Computes the generalized dice score based on inputs passed in to ``update`` previously.

        Return:
            The shape of the returned tensor depends on the ``average`` parameter:

            - If ``average == 'samples'``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(N,)``, where ``N`` stands
                for the number of samples
        """
        tp, fp, _, fn = self._get_final_stats()
        return _generalized_dice_compute(
            tp,
            fp,
            fn,
            average=self.average,
            ignore_index=None if self.reduce is None else self.ignore_index,
            weight_type=self.weight_type,
            zero_division=self.zero_division,
        )
