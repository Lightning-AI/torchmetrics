"""This code is inspired by
https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/vif.py and
https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/full_ref.py#L357.

Reference: https://ieeexplore.ieee.org/abstract/document/1576816
"""
from typing import Any, Optional, Tuple

from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.image.vif import visual_information_fidelity


class VIF(Metric):
    def __init__(self, sigma_n_sq: float = 2.0, data_range: Optional[Tuple[float, float]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("vif_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

        self.sigma_n_sq = sigma_n_sq
        self.data_range = data_range

    def update(self, preds: Tensor, target: Tensor) -> None:
        vif_score = visual_information_fidelity(
            preds=preds, target=target, data_range=self.data_range, sigma_n_sq=self.sigma_n_sq
        )
        self.vif_score += vif_score
        self.total += preds.size(0)

    def compute(self) -> Any:
        return self.vif_score / self.total
