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
from typing import Any, List, Tuple

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

    Formula taken from here: `Aggregate the statistics from multiple devices`_
    """
    # assert len(means_x) > 1 and len(means_y) > 1 and len(vars_x) > 1 and len(vars_y) > 1 and len(corrs_xy) > 1
    mx1, my1, vx1, vy1, cxy1, n1 = means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]
        nb = n1 + n2
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb

        # var_x
        element_x1 = (n1 + 1) * mean_x - n1 * mx1
        vx1 += (element_x1 - mx1) * (element_x1 - mean_x) - (element_x1 - mean_x) ** 2
        element_x2 = (n2 + 1) * mean_x - n2 * mx2
        vx2 += (element_x2 - mx2) * (element_x2 - mean_x) - (element_x2 - mean_x) ** 2
        var_x = vx1 + vx2

        # var_y
        element_y1 = (n1 + 1) * mean_y - n1 * my1
        vy1 += (element_y1 - my1) * (element_y1 - mean_y) - (element_y1 - mean_y) ** 2
        element_y2 = (n2 + 1) * mean_y - n2 * my2
        vy2 += (element_y2 - my2) * (element_y2 - mean_y) - (element_y2 - mean_y) ** 2
        var_y = vy1 + vy2

        # corr
        cxy1 += (element_x1 - mx1) * (element_y1 - mean_y) - (element_x1 - mean_x) * (element_y1 - mean_y)
        cxy2 += (element_x2 - mx2) * (element_y2 - mean_y) - (element_x2 - mean_x) * (element_y2 - mean_y)
        corr_xy = cxy1 + cxy2

        mx1, my1, vx1, vy1, cxy1, n1 = mean_x, mean_y, var_x, var_y, corr_xy, nb
    return var_x, var_y, corr_xy, nb


class PearsonCorrCoef(Metric):
    r"""Computes `Pearson Correlation Coefficient`_:

    .. math::
        P_{corr}(x,y) = \frac{cov(x,y)}{\sigma_x \sigma_y}

    Where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    Forward accepts

    - ``preds`` (float tensor): ``(N,)``
    - ``target``(float tensor): ``(N,)``

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import PearsonCorrCoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson = PearsonCorrCoef()
        >>> pearson(preds, target)
        tensor(0.9849)

    """
    is_differentiable = True
    higher_is_better = None  # both -1 and 1 are optimal
    full_state_update: bool = True
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("mean_x", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("mean_y", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("var_x", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("var_y", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("corr_xy", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("n_total", default=torch.tensor(0.0), dist_reduce_fx=None)

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
