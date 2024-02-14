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
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.roc import BinaryROC, MulticlassROC, MultilabelROC
from torchmetrics.functional.classification.logauc import (
    _binary_logauc_compute,
    _validate_fpr_range
)
from torch import Tensor
from typing import Tuple, Optional, Union, Any

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
        validate_args: bool = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args, **kwargs)
        _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range

    def compute(self) -> Tensor:
        fpr, tpr, _ = super().compute()
        return _binary_logauc_compute(fpr, tpr, fpr_range=self.fpr_range)


class MultiClassLogAUC(MulticlassROC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    pass


class MultiLabelLogAUC(MultilabelROC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    pass


class LogAUC(_ClassificationTaskWrapper):
    pass
