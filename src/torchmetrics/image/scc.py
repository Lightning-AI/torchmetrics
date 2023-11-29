from typing import Any, Optional
from typing_extensions import Literal

import torch
from torch import Tensor, tensor

from torchmetrics.functional.image.scc import _scc_per_channel_compute as _scc_compute, _scc_update
from torchmetrics.metric import Metric

class SpatialCorrelationCoefficient(Metric):
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