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
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor

from torchmetrics.functional.regression.spearman import _spearman_corrcoef_compute, _spearman_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class SpearmanCorrcoef(Metric):
    r"""
    Computes `spearmans rank correlation coefficient`_.

    .. math:
        r_s = = \frac{cov(rg_x, rg_y)}{\sigma_{rg_x} * \sigma_{rg_y}}

    where rg_x and rg_y are the rank associated to the variables x and y. Spearmans correlations coefficient
    corresponds to the standard pearsons correlation coefficient calculated on the rank variables.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Example:
        >>> from torchmetrics import SpearmanCorrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> spearman = SpearmanCorrcoef()
        >>> spearman(preds, target)
        tensor(1.0000)

    """
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        rank_zero_warn(
            "Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _spearman_corrcoef_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes spearmans correlation coefficient."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spearman_corrcoef_compute(preds, target)

    @property
    def is_differentiable(self) -> bool:
        return False
