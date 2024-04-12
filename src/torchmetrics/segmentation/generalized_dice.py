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
from typing_extensions import Literal

from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.generalized_dice import (
    _binary_generalized_dice_score_arg_validation,
    _generalized_dice_reduce,
    _multiclass_generalized_dice_score_arg_validation,
    _multilabel_generalized_dice_score_arg_validation,
)
from torchmetrics.metric import Metric


class BinaryGeneralizedDiceScore(BinaryStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        weight_type: Optional[Literal["square", "simple"]] = "square",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _binary_generalized_dice_score_arg_validation(weight_type, threshold, multidim_average, ignore_index)
        self.validate_args = validate_args
        self.weight_type = weight_type

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _generalized_dice_reduce(
            tp, fp, tn, fn, self.weight_type, average="binary", multidim_average=self.multidim_average
        )


class MulticlassGeneralizedDiceScore(MulticlassStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        weight_type: Optional[Literal["square", "simple"]] = "square",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multiclass_generalized_dice_score_arg_validation(
                weight_type, num_classes, top_k, average, multidim_average, ignore_index
            )
        self.validate_args = validate_args
        self.weight_type = weight_type

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _generalized_dice_reduce(
            tp, fp, tn, fn, self.weight_type, average=self.average, multidim_average=self.multidim_average
        )


class MultilabelGeneralizedDiceScore(MultilabelStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        weight_type: Optional[Literal["square", "simple"]] = "square",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multilabel_generalized_dice_score_arg_validation(
                weight_type, num_labels, threshold, average, multidim_average, ignore_index
            )
        self.validate_args = validate_args
        self.weight_type = weight_type

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _generalized_dice_reduce(
            tp, fp, tn, fn, self.weight_type, average=self.average, multidim_average=self.multidim_average
        )


class GeneralizedDiceScore:
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
        >>> from torch import tensor
        >>> from torchmetrics import GeneralizedDiceScore
        >>> preds = tensor([2, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> generalized_dice_score = GeneralizedDiceScore(num_classes=3)
        >>> generalized_dice_score(preds, target)
        tensor(0.3478)

    """

    def __new__(
        cls,
        num_classes: Optional[int] = None,
        beta: float = 1.0,
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
        assert multidim_average is not None
        kwargs.update({
            "multidim_average": multidim_average,
            "ignore_index": ignore_index,
            "validate_args": validate_args,
        })
        if task == "binary":
            return BinaryGeneralizedDiceScore(beta, threshold, **kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return MulticlassGeneralizedDiceScore(beta, num_classes, top_k, average, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelGeneralizedDiceScore(beta, num_labels, threshold, average, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
