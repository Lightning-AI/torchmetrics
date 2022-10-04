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
from typing import Any, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torchmetrics.functional.classification.auroc import (
    _auroc_compute,
    _auroc_update,
    _binary_auroc_arg_validation,
    _binary_auroc_compute,
    _multiclass_auroc_arg_validation,
    _multiclass_auroc_compute,
    _multilabel_auroc_arg_validation,
    _multilabel_auroc_compute,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.imports import _TORCH_LOWER_1_6


class BinaryAUROC(BinaryPrecisionRecallCurve):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for binary tasks. The AUROC
    score summarizes the ROC curve into an single number that describes the performance of a model for multiple
    thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        max_fpr: If not ``None``, calculates standardized partial AUC over the range ``[0, max_fpr]``.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        A single scalar with the auroc score

    Example:
        >>> from torchmetrics.classification import BinaryAUROC
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryAUROC(thresholds=None)
        >>> metric(preds, target)
        tensor(0.5000)
        >>> metric = BinaryAUROC(thresholds=5)
        >>> metric(preds, target)
        tensor(0.5000)
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        max_fpr: Optional[float] = None,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _binary_auroc_arg_validation(max_fpr, thresholds, ignore_index)
        self.max_fpr = max_fpr

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_auroc_compute(state, self.thresholds, self.max_fpr)


class MulticlassAUROC(MulticlassPrecisionRecallCurve):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for multiclass tasks. The AUROC
    score summarizes the ROC curve into an single number that describes the performance of a model for multiple
    thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``weighted``: Calculates score for each class and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates score for each class and applies no reduction

        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with auroc score per class.
        If `average="macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from torchmetrics.classification import MulticlassAUROC
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassAUROC(num_classes=5, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.5333)
        >>> metric = MulticlassAUROC(num_classes=5, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([1.0000, 1.0000, 0.3333, 0.3333, 0.0000])
        >>> metric = MulticlassAUROC(num_classes=5, average="macro", thresholds=5)
        >>> metric(preds, target)
        tensor(0.5333)
        >>> metric = MulticlassAUROC(num_classes=5, average=None, thresholds=5)
        >>> metric(preds, target)
        tensor([1.0000, 1.0000, 0.3333, 0.3333, 0.0000])
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_auroc_arg_validation(num_classes, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multiclass_auroc_compute(state, self.num_classes, self.average, self.thresholds)


class MultilabelAUROC(MultilabelPrecisionRecallCurve):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for multilabel tasks. The AUROC
    score summarizes the ROC curve into an single number that describes the performance of a model for multiple
    thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifing the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: Calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with auroc score per class.
        If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from torchmetrics.classification import MultilabelAUROC
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> metric = MultilabelAUROC(num_labels=3, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.6528)
        >>> metric = MultilabelAUROC(num_labels=3, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([0.6250, 0.5000, 0.8333])
        >>> metric = MultilabelAUROC(num_labels=3, average="macro", thresholds=5)
        >>> metric(preds, target)
        tensor(0.6528)
        >>> metric = MultilabelAUROC(num_labels=3, average=None, thresholds=5)
        >>> metric(preds, target)
        tensor([0.6250, 0.5000, 0.8333])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multilabel_auroc_arg_validation(num_labels, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multilabel_auroc_compute(state, self.num_labels, self.average, self.thresholds, self.ignore_index)


class AUROC(Metric):
    r"""Area Under the Receiver Operating Characteristic Curve.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.


    Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).
    Works for both binary, multilabel and multiclass problems. In the case of
    multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    For non-binary input, if the ``preds`` and ``target`` tensor have the same
    size the input will be interpretated as multilabel and if ``preds`` have one
    dimension more than the ``target`` tensor the input will be interpretated as
    multiclass.

    .. note::
        If either the positive class or negative class is completly missing in the target tensor,
        the auroc score is meaningless in this case and a score of 0 will be returned together
        with an warning.

    Args:
        num_classes: integer with number of classes for multi-label and multiclass problems.

            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            - ``'micro'`` computes metric globally. Only works for multilabel problems
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``None``, ``"macro"`` or ``"weighted"``.
        ValueError:
            If ``max_fpr`` is not a ``float`` in the range ``(0, 1]``.
        RuntimeError:
            If ``PyTorch version`` is ``below 1.6`` since ``max_fpr`` requires ``torch.bucketize``
            which is not available below 1.6.
        ValueError:
            If the mode of data (binary, multi-label, multi-class) changes between batches.

    Example (binary case):
        >>> from torchmetrics import AUROC
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> auroc = AUROC(pos_label=1)
        >>> auroc(preds, target)
        tensor(0.5000)

    Example (multiclass case):
        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> auroc = AUROC(num_classes=3)
        >>> auroc(preds, target)
        tensor(0.7778)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __new__(
        cls,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        max_fpr: Optional[float] = None,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryAUROC(max_fpr, **kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassAUROC(num_classes, average, **kwargs)
            if task == "multilabel":
                assert isinstance(num_labels, int)
                return MultilabelAUROC(num_labels, average, **kwargs)
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
        pos_label: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        max_fpr: Optional[float] = None,
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, "macro", "weighted", "micro")
        if self.average not in allowed_average:
            raise ValueError(
                f"Argument `average` expected to be one of the following: {allowed_average} but got {average}"
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

            if _TORCH_LOWER_1_6:
                raise RuntimeError(
                    "`max_fpr` argument requires `torch.bucketize` which is not available below PyTorch version 1.6"
                )

        self.mode: DataType = None  # type: ignore
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AUROC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        preds, target, mode = _auroc_update(preds, target)

        self.preds.append(preds)
        self.target.append(target)

        if self.mode and self.mode != mode:
            raise ValueError(
                "The mode of data (binary, multi-label, multi-class) should be constant, but changed"
                f" between batches from {self.mode} to {mode}"
            )
        self.mode = mode

    def compute(self) -> Tensor:
        """Computes AUROC based on inputs passed in to ``update`` previously."""
        if not self.mode:
            raise RuntimeError("You have to have determined mode.")
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _auroc_compute(
            preds,
            target,
            self.mode,
            self.num_classes,
            self.pos_label,
            self.average,
            self.max_fpr,
        )
