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
from torch import Tensor

from torchmetrics.functional.regression.concordance import _concordance_corrcoef_compute
from torchmetrics.regression.pearson import PearsonCorrCoef, _final_aggregation


class ConcordanceCorrCoef(PearsonCorrCoef):
    r"""Computes concordance correlation coefficient that measures the agreement between two variables. It is
    defined as.

    .. math::
        \rho_c = \frac{2 \rho \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}

    where :math:`\mu_x, \mu_y` is the means for the two variables, :math:`\sigma_x^2, \sigma_y^2` are the corresponding
    variances and \rho is the pearson correlation coefficient between the two variables.

    Forward accepts
    - ``preds`` (float tensor): either single output tensor with shape ``(N,)`` or multioutput tensor of shape ``(N,d)``
    - ``target``(float tensor): either single output tensor with shape ``(N,)`` or multioutput tensor of shape ``(N,d)``

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torchmetrics import ConcordanceCorrCoef
        >>> import torch
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> concordance = ConcordanceCorrCoef()
        >>> concordance(preds, target)
        tensor(0.9777)

    Example (multi output regression):
        >>> from torchmetrics import ConcordanceCorrCoef
        >>> import torch
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> concordance = ConcordanceCorrCoef(num_outputs=2)
        >>> concordance(preds, target)
        tensor([0.7273, 0.9887])
    """

    def compute(self) -> Tensor:
        """Computes final concordance correlation coefficient over metric states."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            mean_x, mean_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
            )
        else:
            mean_x = self.mean_x
            mean_y = self.mean_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _concordance_corrcoef_compute(mean_x, mean_y, var_x, var_y, corr_xy, n_total)
