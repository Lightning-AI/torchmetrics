"""This code is inspired by
https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/vif.py and
https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/full_ref.py#L357.

Reference: https://ieeexplore.ieee.org/abstract/document/1576816
"""
from typing import Any

from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.image.vif import visual_information_fidelity


class VIF(Metric):
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

    def __init__(
            self, sigma_n_sq: float = 2.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("vif_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

        self.sigma_n_sq = sigma_n_sq

    def update(self, preds: Tensor, target: Tensor) -> None:
        vif_score = visual_information_fidelity(
            preds=preds, target=target, sigma_n_sq=self.sigma_n_sq
        )
        self.vif_score += vif_score
        self.total += preds.size(0)

    def compute(self) -> Any:
        return self.vif_score / self.total
