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

from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric


def _final_aggregation(mxs, mys, vxs, vys, cxys, ns):
    """
    Aggregate the statistics from multiple devices. Formula taken from here:
    https://stackoverflow.com/questions/68395368/estimate-running-correlation-on-multiple-nodes
    """
    mx1, my1, vx1, vy1, cxy1, n1 = mxs[0], mys[0], vxs[0], vys[0], cxys[0], ns[0]
    for i in range(1, len(mxs)):
        mx2, my2, vx2, vy2, cxy2, n2 = mxs[i], mys[i], vxs[i], vys[i], cxys[i], ns[i]

        n = n1 + n2
        mx = (n1 * mx1 + n2 * mx2) / n
        my = (n1 * my1 + n2 * my2) / n
        vx = n1 * vx1 + n1 * (mx1 - mx) * (my1 - my) + n2 * vx2 + n2 * (mx2 - mx) * (my2 - my)
        vy = n1 * vy1 + n1 * (my1 - my) * (my1 - my) + n2 * vy2 + n2 * (my2 - my) * (my2 - my)
        cxy = n1 * cxy1 + n1 * (mx1 - mx) * (my1 - my) + n2 * cxy2 + n2 * (mx2 - mx) * (my2 - my)

        mx1, my1, vx1, vy1, cxy1, n1 = mx, my, vx, vy, cxy, n

    return vx, vy, cxy, n


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
    preds: List[Tensor]
    target: List[Tensor]

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

        self.add_state("mx", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("my", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("vx", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("vy", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("cxy", default=torch.zeros(1), dist_reduce_fx=None)
        self.add_state("n", default=torch.zeros(1), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.mx, self.my, self.vx, self.vy, self.cxy, self.n = _pearson_corrcoef_update(
            preds, target, self.mx, self.my, self.vx, self.vy, self.cxy, self.n
        )

    def compute(self) -> Tensor:
        """
        Computes pearson correlation coefficient over state.
        """
        if isinstance(self.mx, list):  # reduce over multiple devices, need further reduction
            vx, vy, cxy, n = _final_aggregation(self.mx, self.my, self.vx, self.vy, self.cxy, self.n)
        else:
            vx = self.vx
            vy = self.vy
            cxy = self.cxy
            n = self.n

        return _pearson_corrcoef_compute(vx, vy, cxy, n)

    @property
    def is_differentiable(self) -> bool:
        return True
