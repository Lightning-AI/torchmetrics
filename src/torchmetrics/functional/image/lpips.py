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

# Content copied from
# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# and
# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
# and with adjustments from
# https://github.com/richzhang/PerceptualSimilarity/pull/114/files
# due to package no longer being maintained
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# All rights reserved.
# License under BSD 2-clause
import inspect
import os
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from typing_extensions import Literal

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13

_weight_map = {
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
}

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["learned_perceptual_image_patch_similarity"]
else:
    from torchvision import models as tv


def _get_net(net: str, pretrained: bool) -> nn.modules.container.Sequential:
    """Get torchvision network.

    Args:
        net: Name of network
        pretrained: If pretrained weights should be used

    """
    if _TORCHVISION_GREATER_EQUAL_0_13:
        if pretrained:
            pretrained_features = getattr(tv, net)(weights=getattr(tv, _weight_map[net]).IMAGENET1K_V1).features
        else:
            pretrained_features = getattr(tv, net)(weights=None).features
    else:
        pretrained_features = getattr(tv, net)(pretrained=pretrained).features
    return pretrained_features


class SqueezeNet(torch.nn.Module):
    """SqueezeNet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        pretrained_features = _get_net("squeezenet1_1", pretrained)

        self.N_slices = 7
        slices = []
        feature_ranges = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]
        for feature_range in feature_ranges:
            seq = torch.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)

        self.slices = nn.ModuleList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""
        squeeze_output = namedtuple("squeeze_output", ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"])

        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return squeeze_output(*relus)


class Alexnet(torch.nn.Module):
    """Alexnet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        alexnet_pretrained_features = _get_net("alexnet", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("alexnet_outputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        return alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class Vgg16(torch.nn.Module):
    """Vgg16 implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        vgg_pretrained_features = _get_net("vgg16", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("vgg_outputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def spatial_average(in_tens: Tensor, keepdim: bool = True) -> Tensor:
    """Spatial averaging over heigh and width of images."""
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsam(in_tens: Tensor, out_hw: Tuple[int, int] = (64, 64)) -> Tensor:
    """Upsample input with bilinear interpolation."""
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(in_tens)


def normalize_tensor(in_feat: Tensor, eps: float = 1e-10) -> Tensor:
    """Normalize tensors."""
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class ScalingLayer(nn.Module):
    """Scaling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: Tensor) -> Tensor:
        """Process input."""
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [nn.Dropout()] if use_dropout else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),  # type: ignore[list-item]
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Process input."""
        return self.model(x)


class _LPIPS(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        spatial: bool = False,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = True,
        model_path: Optional[str] = None,
        eval_mode: bool = True,
    ) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            pretrained: This flag controls the linear layers should be pretrained version or random
            net: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            spatial: If input should be spatial averaged
            pnet_rand: If backbone should be random or use imagenet pre-trained weights
            pnet_tune: If backprop should be enabled
            use_dropout: If dropout layers should be added
            model_path: Model path to load pretained models from
            eval_mode: If network should be in evaluation mode
        """
        super().__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = Alexnet  # type: ignore[assignment]
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = SqueezeNet  # type: ignore[assignment]
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)  # type: ignore[assignment]

        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(
                    os.path.join(inspect.getfile(self.__init__), "..", f"lpips_models/{net}.pth")  # type: ignore[misc]
                )

            self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

        if eval_mode:
            self.eval()

    def forward(
        self, in0: Tensor, in1: Tensor, retperlayer: bool = False, normalize: bool = False
    ) -> Union[int, Tuple[int, List[Tensor]]]:
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.spatial:
            res = [
                upsam(self.lins[kk](diffs[kk]), out_hw=in0.shape[2:]) for kk in range(self.L)  # type: ignore[arg-type]
            ]
        else:
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]

        val = 0
        for layer in range(self.L):
            val += res[layer]  # type: ignore[assignment]

        if retperlayer:
            return (val, res)
        return val


class _NoTrainLpips(_LPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainLpips":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor, normalize: bool) -> bool:
    """Check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check  # type: ignore[return-value]


def _lpips_update(img1: Tensor, img2: Tensor, net: nn.Module, normalize: bool) -> Tuple[Tensor, Union[int, Tensor]]:
    if not (_valid_img(img1, normalize) and _valid_img(img2, normalize)):
        raise ValueError(
            "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
            f" Got input with shape {img1.shape} and {img2.shape} and values in range"
            f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
            f" expected to be in the {[0,1] if normalize else [-1,1]} range."
        )
    loss = net(img1, img2, normalize=normalize).squeeze()
    return loss, img1.shape[0]


def _lpips_compute(sum_scores: Tensor, total: Union[Tensor, int], reduction: Literal["sum", "mean"] = "mean") -> Tensor:
    return sum_scores / total if reduction == "mean" else sum_scores


def learned_perceptual_image_patch_similarity(
    img1: Tensor,
    img2: Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    reduction: Literal["sum", "mean"] = "mean",
    normalize: bool = False,
) -> Tensor:
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates the perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    Args:
        img1: first set of images
        img2: second set of images
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
        tensor(0.1008, grad_fn=<DivBackward0>)

    """
    net = _NoTrainLpips(net=net_type)
    loss, total = _lpips_update(img1, img2, net, normalize)
    return _lpips_compute(loss.sum(), total, reduction)
