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


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """Computes total variation loss.

    Adapted from https://github.com/jxgu1016/Total_Variation_Loss.pytorch
    Args:
        img (torch.Tensor): A NCHW image batch.

    Returns:
        A loss scalar value.
    """

    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    _height = img.size()[2]
    _width = img.size()[3]
    _count_height = tensor_size(img[:, :, 1:, :])
    _count_width = tensor_size(img[:, :, :, 1:])
    _height_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : _height - 1, :]), 2).sum()
    _width_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : _width - 1]), 2).sum()
    return (2 * (_height_tv / _count_height + _width_tv / _count_width)) / img.size()[0]
