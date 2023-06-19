# Copyright The Lightning team.
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
from typing import Any

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.nominal.fleiss_kappa import _fleiss_kappa_compute, _fleiss_kappa_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class FleissKappa(Metric):
    """Computes Fleiss' kappa for evaluating the degree of agreement between raters."""

    def __init__(self, mode: Literal["counts", "probs"] = "counts", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if mode not in ["counts", "probs"]:
            raise ValueError("Argument ``mode`` must be one of 'counts' or 'probs'.")
        self.add_state("counts", default=[], dist_reduce_fx="cat")

    def update(self, ratings: Tensor) -> None:
        """Updates the counts for fleiss kappa metric."""
        counts = _fleiss_kappa_update(ratings, self.mode)
        self.counts.append(counts)

    def compute(self) -> Tensor:
        """Computes Fleiss' kappa."""
        counts = dim_zero_cat(self.counts)
        return _fleiss_kappa_compute(counts)
