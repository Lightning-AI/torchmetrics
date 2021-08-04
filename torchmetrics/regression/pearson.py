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
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    corrs_xy: Tensor,
    nbs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here:
    https://stackoverflow.com/questions/68395368/estimate-running-correlation-on-multiple-nodes
    """
    # assert len(means_x) > 1 and len(means_y) > 1 and len(vars_x) > 1 and len(vars_y) > 1 and len(corrs_xy) > 1
    mx1, my1, vx1, vy1, cxy1, n1 = means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]

        nb = n1 + n2
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb
        var_x = 1 / (n1 + n2 - 1) * ((n1 - 1) * vx1 + (n2 - 1) * vx2 + ((n1 * n2) / (n1 + n2)) * (mx1 - mx2) ** 2)
        var_y = 1 / (n1 + n2 - 1) * ((n1 - 1) * vy1 + (n2 - 1) * vy2 + ((n1 * n2) / (n1 + n2)) * (my1 - my2) ** 2)

        corr1 = n1 * cxy1 + n1 * (mx1 - mean_x) * (my1 - mean_y)
        corr2 = n2 * cxy2 + n2 * (mx2 - mean_x) * (my2 - mean_y)
        corr_xy = (corr1 + corr2) / (n1 + n2)

        mx1, my1, vx1, vy1, cxy1, n1 = mean_x, mean_y, var_x, var_y, corr_xy, nb

    return var_x, var_y, corr_xy, nb


class PearsonCorrcoef(Metric):
    r"""
    Computes `pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_:

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
    preds: List[Tensor]
    target: List[Tensor]
    mean_x: Tensor
    mean_y: Tensor
    var_x: Tensor
    var_y: Tensor
    corr_xy: Tensor
    n_total: Tensor

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("mean_x", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("mean_y", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("var_x", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("var_y", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("corr_xy", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("n_total", default=torch.zeros(1), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total = _pearson_corrcoef_update(
            preds, target, self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
        )

    def compute(self) -> Tensor:
        """Computes pearson correlation coefficient over state."""
        if self.mean_x.numel() > 1:  # multiple devices, need further reduction
            var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
            )
        else:
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total

        return _pearson_corrcoef_compute(var_x, var_y, corr_xy, n_total)

    @property
    def is_differentiable(self) -> bool:
        return True
