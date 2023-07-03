from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.image.vif import _vif_per_channel
from torchmetrics.utilities.distributed import reduce


class VisualInformationFidelity(Metric):
    """Compute Pixel Based Visual Information Fidelity (vif-p).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``vif-p`` (:class:`~torch.Tensor`): Tensor with vif-p score

    Args:
        sigma_n_sq: variance of the visual noise
    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, sigma_n_sq: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("vif_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.sigma_n_sq = sigma_n_sq

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        batches, channels = preds.size(0), preds.size(1)
        vif_per_channel = [
            _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], self.sigma_n_sq) for i in range(channels)
        ]
        vif_per_channel = torch.mean(torch.stack(vif_per_channel), 0) if channels > 1 else torch.cat(vif_per_channel)
        self.vif_score += torch.sum(vif_per_channel)
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute vif-p over state."""
        return self.vif_score / self.total
