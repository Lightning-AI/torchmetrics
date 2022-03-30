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

from typing import Any, Dict, List, Optional, Sequence


from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.scc import _scc_compute, _scc_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat

class SpatialCorrelationCoefficient(Metric):
    preds: List[Tensor]
    target: List[Tensor]
    higher_is_better: bool = True

    def __init__(
        self,
        kernel_size: Sequence[int] = (9, 9),
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        rank_zero_warn(
            "Metric `SpatialCorrelationCoefficient` will save all target and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.kernel_size = kernel_size
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and target.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _scc_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _scc_compute(preds, target, self.kernel_size, self.reduction)

