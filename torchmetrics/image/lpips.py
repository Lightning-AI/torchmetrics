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
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor


from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

if _LPIPS_AVAILABLE:
    from lpips import LPIPS as Lpips_net
else:
    class Lpips_net(torch.nn.Module):  # type: ignore
        pass


class NoTrainLpips(Lpips_net):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)


class LPIPS(Metric):
    r"""
   
    """
    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        net_type: str = 'alex',
        reduction: str = 'mean',
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not _LPIPS_AVAILABLE:
            raise ValueError(
                "LPIPS metric requires that lpips is installed."
                "Either install as `pip install torchmetrics[image]` or `pip install lpips`"
            )

        valid_net_type = ('vgg', 'vgg16', 'alex', 'squeeze')
        if net_type not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}.")
        self.net = NoTrainLpips(net=net_type)

        valid_reduction = ('mean', 'sum')
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self._reduction = reduction

        self.add_state("sum_scores", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        loss = self.net(img1, img2)
        self.sum_scores += loss
        self.total += img1.shape[0]

    def compute(self) -> Tensor:
        if self.reduction == 'mean':
            return self.sum_scores / self.total
        elif self.reduction == 'sum':
            return self.sum_scores

