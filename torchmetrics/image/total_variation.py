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

import torch

from torchmetrics.metric import Metric


class TotalVariation(Metric):
    """Computes Total Variation loss.

    Adapted from: https://github.com/jxgu1016/Total_Variation_Loss.pytorch
    Args:
        dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        compute_on_step: Forward only calls ``update()`` and returns None if this is set to
            False.
    """

    is_differentiable = True
    higher_is_better = False
    current: torch.Tensor
    total: torch.Tensor

    def __init__(self, dist_sync_on_step: bool = False, compute_on_step: bool = True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("current", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, img: torch.Tensor) -> None:
        """Update method for TV Loss.

        Args:
            img (torch.Tensor): A NCHW image batch.

        Returns:
            A loss scalar value.
        """
        _height = img.size()[2]
        _width = img.size()[3]
        _count_height = self.tensor_size(img[:, :, 1:, :])
        _count_width = self.tensor_size(img[:, :, :, 1:])
        _height_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : _height - 1, :]), 2).sum()
        _width_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : _width - 1]), 2).sum()
        self.current += 2 * (_height_tv / _count_height + _width_tv / _count_width)
        self.total += img.numel()

    def compute(self):
        return self.current.float() / self.total

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
