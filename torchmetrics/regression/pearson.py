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

import torch
from torch import Tensor

from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class PearsonCorrcoef(Metric):
    r"""
    Computes `pearson correlation coefficient
    <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_:

    .. math::
        P_{corr}(x,y) = \frac{cov(x,y)}{\sigma_x \sigma_y}

    Where :math:`y` is a tensor of target values, and :math:`x` is a
    tensor of predictions.

    Forward accepts

    - ``preds`` (float tensor): ``(N,)``
    - ``target``(float tensor): ``(N,)``

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:
        >>> from torchmetrics import PearsonCorrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson = PearsonCorrcoef()
        >>> pearson(preds, target)
        tensor(0.9849)

    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        rank_zero_warn(
            'Metric `PearsonCorrcoef` will save all targets and predictions in buffer.'
            ' For large datasets this may lead to large memory footprint.'
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _pearson_corrcoef_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        """
        Computes pearson correlation coefficient over state.
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _pearson_corrcoef_compute(preds, target)

    @property
    def is_differentiable(self):
        return True
