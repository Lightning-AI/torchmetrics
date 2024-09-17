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
from typing import Any, List, Optional, Tuple, Type, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.roc import BinaryROC, MulticlassROC, MultilabelROC
from torchmetrics.functional.classification.logauc import (
    _binary_logauc_compute,
    _multiclass_logauc_compute,
    _validate_fpr_range,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask


class BinaryLogAUC(BinaryROC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        fpr_range: Tuple[float, float] = (0.001, 0.1),
        thresholds: Optional[Union[float, Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args, **kwargs)
        _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range

    def compute(self) -> Tensor:
        """Computes the log AUC score."""
        fpr, tpr, _ = super().compute()
        return _binary_logauc_compute(fpr, tpr, fpr_range=self.fpr_range)


class MulticlassLogAUC(MulticlassROC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        fpr_range: Tuple[float, float] = (0.001, 0.1),
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        average: Optional[Literal["macro", "none"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            thresholds=thresholds,
            average=None,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range
        self.average = average

    def compute(self) -> Tensor:
        """Computes the log AUC score."""
        fpr, tpr, _ = super().compute()
        return _multiclass_logauc_compute(fpr, tpr, fpr_range=self.fpr_range, average=self.average)


class MultilabelLogAUC(MultilabelROC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        fpr_range: Tuple[float, float] = (0.001, 0.1),
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range


class LogAUC(_ClassificationTaskWrapper):
    def __new__(  # type: ignore[misc]
        cls: Type["LogAUC"],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        fp_range: Optional[Tuple[float, float]] = (0.001, 0.1),
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({
            "thresholds": thresholds,
            "fp_range": fp_range,
            "ignore_index": ignore_index,
            "validate_args": validate_args,
        })
        if task == ClassificationTask.BINARY:
            return BinaryLogAUC(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassLogAUC(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelLogAUC(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")
