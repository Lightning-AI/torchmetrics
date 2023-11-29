from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.functional.image.scc import _scc_per_channel_compute as _scc_compute, _scc_update
from torchmetrics.metric import Metric

class SpatialCorrelationCoefficient(Metric):
    """Compute Spatial Correlation Coefficient (SCC_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)`` or ``(N,H,W)``.
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)`` or ``(N,H,W)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``scc`` (:class:`~torch.Tensor`): Tensor with scc score

    Args:
        hp_filter: High-pass filter tensor. default: tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).
        window_size: Local window size integer. default: 8.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.image import SpatialCorrelationCoefficient as SCC
        >>> preds = torch.randn([32, 3, 64, 64])
        >>> target = torch.randn([32, 3, 64, 64])
        >>> scc = SCC()
        >>> scc(preds, target)
        tensor(0.0022)

    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    scc_score: Tensor
    total: Tensor

    def __init__(self, high_pass_filter: Tensor = tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]), 
                 window_size: int = 11, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        self.hp_filter = high_pass_filter
        self.ws = window_size
        
        self.add_state("scc_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target, hp_filter = _scc_update(preds, target, self.hp_filter, self.ws)
        scc_per_channel = [
            _scc_compute(preds[:, i, :, :].unsqueeze(1), target[:, i, :, :].unsqueeze(1), hp_filter, self.ws)
            for i in range(preds.size(1))
            ]
        self.scc_score += torch.sum(torch.mean(torch.cat(scc_per_channel, dim=1), dim=[1,2,3]))
        self.total += preds.size(0)
    
    def compute(self) -> Tensor:
        """Compute the VIF score based on inputs passed in to ``update`` previously."""
        return self.scc_score / self.total