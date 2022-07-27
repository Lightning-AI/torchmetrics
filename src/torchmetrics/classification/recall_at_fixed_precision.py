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

from torch import Tensor

from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torchmetrics.functional.classification.recall_at_fixed_precision import (
    _binary_recall_at_fixed_precision_arg_validation,
    _binary_recall_at_fixed_precision_compute,
    _multiclass_recall_at_fixed_precision_arg_compute,
    _multiclass_recall_at_fixed_precision_arg_validation,
    _multilabel_recall_at_fixed_precision_arg_compute,
    _multilabel_recall_at_fixed_precision_arg_validation,
)
from torchmetrics.utilities.data import dim_zero_cat


class BinaryRecallAtFixedPrecision(BinaryPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        min_precision: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = 100,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds, ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _binary_recall_at_fixed_precision_arg_validation(min_precision, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_precision = min_precision

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_recall_at_fixed_precision_compute(state, self.thresholds, self.min_precision)


class MulticlassRecallAtFixedPrecision(MulticlassPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        min_precision: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = 100,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_recall_at_fixed_precision_arg_validation(num_classes, min_precision, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_precision = min_precision

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multiclass_recall_at_fixed_precision_arg_compute(
            state, self.num_classes, self.thresholds, self.min_precision
        )


class MultilabelRecallAtFixedPrecision(MultilabelPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        min_precision: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = 100,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multilabel_recall_at_fixed_precision_arg_validation(num_labels, min_precision, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_precision = min_precision

    def compute(self) -> Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _multilabel_recall_at_fixed_precision_arg_compute(
            state, self.num_labels, self.thresholds, self.ignore_index, self.min_precision
        )
