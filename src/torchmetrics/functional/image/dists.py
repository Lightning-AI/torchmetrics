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
#
# Below is a derivative work based on the original work:
# https://github.com/dingkeyan93/DISTS
# with the following license:
#
# MIT License
# Copyright (c) 2020 Keyan Ding
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d
from typing_extensions import Literal

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["deep_image_structure_and_texture_similarity"]

_PATH_WEIGHT_DISTS = Path(__file__).resolve().parent / "dists_models" / "weights.pt"


class L2pooling(nn.Module):
    """L2 pooling layer."""

    filter: Tensor

    def __init__(self, filter_size: int = 5, stride: int = 2, channels: int = 3) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer("filter", g[None, None, :, :].repeat(self.channels, 1, 1, 1))

    def forward(self, tensor: Tensor) -> Tensor:
        """Forward pass of the layer."""
        tensor = tensor**2
        out = conv2d(tensor, self.filter, stride=self.stride, padding=self.padding, groups=tensor.shape[1])
        return (out + 1e-12).sqrt()


class DISTSNetwork(torch.nn.Module):
    """DISTS network."""

    alpha: Tensor
    beta: Tensor
    mean: Tensor
    std: Tensor

    def __init__(self, load_weights: bool = True) -> None:
        super().__init__()

        if _TORCHVISION_AVAILABLE:
            from torchvision import models
        else:
            raise ModuleNotFoundError(
                "DISTS requires torchvision to be installed. Please install it with `pip install torchvision`."
            )

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        if load_weights:
            if not _PATH_WEIGHT_DISTS.exists():
                raise FileNotFoundError(f"The weights file is not found in {_PATH_WEIGHT_DISTS}")
            weights = torch.load(str(_PATH_WEIGHT_DISTS))
            self.alpha.data = weights["alpha"]
            self.beta.data = weights["beta"]

    def forward_once(self, x: Tensor) -> List[Tensor]:
        """Forward pass of the network."""
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x: Tensor, y: Tensor, require_grad: bool = False) -> Tensor:
        """Computes DISTS score between two images."""
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.inference_mode():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1, dist2, c1, c2 = 0, 0, 1e-6, 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            s1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * s1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * s2).sum(1, keepdim=True)

        return 1 - (dist1 + dist2).squeeze()


def _dists_update(preds: Tensor, target: Tensor) -> Tensor:
    dists = DISTSNetwork().to(preds.device)
    return dists(preds, target, require_grad=preds.requires_grad)


def _dists_compute(scores: Tensor, reduction: Optional[Literal["sum", "mean", "none"]]) -> Tensor:
    if reduction == "sum":
        return scores.sum()
    if reduction == "mean":
        return scores.mean()
    if reduction is None or reduction == "none":
        return scores
    raise ValueError(f"Argument {reduction} is not valid. Choose 'sum', 'mean' or 'none'., but got {reduction}")


def deep_image_structure_and_texture_similarity(
    preds: Tensor, target: Tensor, reduction: Optional[Literal["sum", "mean", "none"]] = None
) -> Tensor:
    """Calculates `Deep Image Structure and Texture Similarity`_ (DISTS) score.

    Args:
        preds: Predicted image tensor.
        target: Target image tensor.
        reduction: Reduction method for the output.

    Returns:
        DISTS Similarity score between the two images.

    Example:
        >>> from torch import rand
        >>> preds = rand(5, 3, 256, 256)
        >>> target = rand(5, 3, 256, 256)
        >>> deep_image_structure_and_texture_similarity(preds, target)
        tensor([0.1285, 0.1344, 0.1356, 0.1277, 0.1276], grad_fn=<RsubBackward1>)
        >>> deep_image_structure_and_texture_similarity(preds, target, reduction='mean')
        tensor(0.1308, grad_fn=<MeanBackward0>)

    """
    scores = _dists_update(preds, target)
    return _dists_compute(scores, reduction)
