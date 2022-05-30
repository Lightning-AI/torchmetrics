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
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification_new.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_compute,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_compute,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_compute,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.metric import Metric


class BinaryConfusionMatrix(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        normalize: Optional[str] = None,
        validate_args: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize)
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args

        self.add_state("confmat", torch.zeros(2, 2), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _binary_confusion_matrix_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(preds, target, self.threshold, self.ignore_index)
        confmat = _binary_confusion_matrix_update(preds, target)
        self.confmat += confmat

    def compute(self) -> Tensor:
        return _binary_confusion_matrix_compute(self.confmat, self.normalize)


class MultiClassConfusionMatrix(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        normalize: Optional[str] = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args

        self.add_state("confmat", torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multiclass_confusion_matrix_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index)
        confmat = _multiclass_confusion_matrix_update(preds, target, self.num_classes)
        self.confmat += confmat

    def compute(self) -> Tensor:
        return _multiclass_confusion_matrix_compute(self.confmat, self.normalize)


class MultiLabelConfusionMatrix(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        ignore_index: Optional[int] = None,
        normalize: Optional[str] = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(num_labels, ignore_index, normalize)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args

        self.add_state("confmat", torch.zeros(num_labels, 2, 2), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multilabel_confusion_matrix_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target = _multilabel_confusion_matrix_format(preds, target, self.ignore_index)
        confmat = _multilabel_confusion_matrix_update(preds, target, self.num_labels)
        self.confmat += confmat

    def compute(self) -> Tensor:
        return _multilabel_confusion_matrix_compute(self.confmat, self.normalize)
