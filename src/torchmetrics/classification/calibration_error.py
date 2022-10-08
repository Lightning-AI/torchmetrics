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
from typing import Any, List, Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _binary_calibration_error_tensor_validation,
    _binary_calibration_error_update,
    _binary_confusion_matrix_format,
    _ce_compute,
    _ce_update,
    _multiclass_calibration_error_arg_validation,
    _multiclass_calibration_error_tensor_validation,
    _multiclass_calibration_error_update,
    _multiclass_confusion_matrix_format,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.prints import rank_zero_warn


class BinaryCalibrationError(Metric):
    r"""`Computes the Top-label Calibration Error`_ for binary tasks. The expected calibration error can be used to
    quantify how well a given model is calibrated e.g. how well the predicted output probabilities of the model
    matches the actual probabilities of the ground truth distribution.

    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|, \text{L1 norm (Expected Calibration Error)}

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i), \text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}, \text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import BinaryCalibrationError
        >>> preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
        >>> metric(preds, target)
        tensor(0.2900)
        >>> metric = BinaryCalibrationError(n_bins=2, norm='l2')
        >>> metric(preds, target)
        tensor(0.2918)
        >>> metric = BinaryCalibrationError(n_bins=2, norm='max')
        >>> metric(preds, target)
        tensor(0.3167)
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _binary_calibration_error_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(
            preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False
        )
        confidences, accuracies = _binary_calibration_error_update(preds, target)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)


class MulticlassCalibrationError(Metric):
    r"""`Computes the Top-label Calibration Error`_ for multiclass tasks. The expected calibration error can be used
    to quantify how well a given model is calibrated e.g. how well the predicted output probabilities of the model
    matches the actual probabilities of the ground truth distribution.

    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|, \text{L1 norm (Expected Calibration Error)}

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i), \text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}, \text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import MulticlassCalibrationError
        >>> preds = torch.tensor([[0.25, 0.20, 0.55],
        ...                       [0.55, 0.05, 0.40],
        ...                       [0.10, 0.30, 0.60],
        ...                       [0.90, 0.05, 0.05]])
        >>> target = torch.tensor([0, 1, 2, 0])
        >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l1')
        >>> metric(preds, target)
        tensor(0.2000)
        >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l2')
        >>> metric(preds, target)
        tensor(0.2082)
        >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='max')
        >>> metric(preds, target)
        tensor(0.2333)
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_calibration_error_arg_validation(num_classes, n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multiclass_calibration_error_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(
            preds, target, ignore_index=self.ignore_index, convert_to_labels=False
        )
        confidences, accuracies = _multiclass_calibration_error_update(preds, target)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)


class CalibrationError(Metric):
    r"""Calibration Error.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    `Computes the Top-label Calibration Error`_
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    L1 norm (Expected Calibration Error)

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|

    Infinity norm (Maximum Calibration Error)

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i)

    L2 norm (Root Mean Square Calibration Error)

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`,
    :math:`c_i` is the average confidence of predictions in bin :math:`i`, and
    :math:`b_i` is the fraction of data points in bin :math:`i`.

    .. note::
        L2-norm debiasing is not yet supported.

    Args:
        n_bins: Number of bins to use when computing probabilities and accuracies.
        norm: Norm used to compare empirical and expected probability bins.
            Defaults to "l1", or Expected Calibration Error.
        debias: Applies debiasing term, only implemented for l2 norm. Defaults to True.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    DISTANCES = {"l1", "l2", "max"}
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __new__(
        cls,
        n_bins: int = 15,
        norm: str = "l1",
        task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if task is not None:
            kwargs.update(dict(n_bins=n_bins, norm=norm, ignore_index=ignore_index, validate_args=validate_args))
            if task == "binary":
                return BinaryCalibrationError(**kwargs)
            if task == "multiclass":
                assert isinstance(num_classes, int)
                return MulticlassCalibrationError(num_classes, **kwargs)
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
        n_bins: int = 15,
        norm: str = "l1",
        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        if norm not in self.DISTANCES:
            raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError(f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}")
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Computes top-level confidences and accuracies for the input probabilities and appends them to internal
        state.

        Args:
            preds (Tensor): Model output probabilities.
            target (Tensor): Ground-truth target class labels.
        """
        confidences, accuracies = _ce_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Computes calibration error across all confidences and accuracies.

        Returns:
            Tensor: Calibration error across previously collected examples.
        """
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.bin_boundaries.to(self.device), norm=self.norm)
