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
from typing_extensions import Literal
from torchmetrics import Metric
import torch
from torchmetrics.functional.other.procrustes import procrustes_disparity

class ProcrustesDisparity(Metric):

    def __init__(self, average: Literal["mean", "sum"] = 'mean', **kwargs):
        super().__init__(**kwargs)
        if average not in ("mean", "sum"):
            raise ValueError(f"Argument `average` must be one of ['mean', 'sum'], got {average}")
        self.average = average
        self.add_state("disparity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, dataset1: torch.Tensor, dataset2: torch.Tensor) -> None:
        disparity = procrustes_disparity(dataset1, dataset2)
        self.disparity += disparity.sum()
        self.total += disparity.numel()

    def compute(self) -> torch.Tensor:
        if self.average == "mean":
            return self.disparity / self.total
        return self.disparity

    def plot