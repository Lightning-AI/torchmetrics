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
from torch._C import Value


from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

if _LPIPS_AVAILABLE:
    from lpips import LPIPS as Lpips_net
    from lpips import normalize_tensor, upsample, spatial_average
else:
    class Lpips_net(torch.nn.Module):  # type: ignore
        pass


class NoTrainLpips(Lpips_net):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)

    def forward(self, in0: Tensor, in1: Tensor):
        """ 
        Adjusted from: https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
        Overwritten to make sure the module is scriptable
        """
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for layer in range(1, self.L):
            val += res[layer]

        return val


def _valid_img(img: Tensor) -> bool:
    """ check that input is a valid image to the network """
    return img.ndim == 4 and img.shape[1] == 3 and img.min() >= -1.0 and img.max() <= 1.0


class LPIPS(Metric):
    r"""
   
    """
    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        net_type: str = 'alex',
        reduction: str = 'mean',
        compute_on_step: bool = True,
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

        valid_net_type = ('vgg', 'alex', 'squeeze')
        if net_type not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}.")
        self.net = NoTrainLpips(net=net_type)

        valid_reduction = ('mean', 'sum')
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        self.add_state("sum_scores", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        if not (_valid_img(img1) and _valid_img(img2)):
            raise ValueError("Expected both input arguments to be normalized tensors (all values in range [-1,1])"
                             f" and to have shape [N, 3, H, W] but `img1` have shape {img1.shape} with values in"
                             f" range {[img1.min(), img1.max()]} and `img2` have shape {img2.shape} with value"
                             f" in range {[img2.min(), img2.max()]}")

        loss = self.net(img1, img2).squeeze()
        self.sum_scores += loss.sum()
        self.total += img1.shape[0]

    def compute(self) -> Tensor:
        if self.reduction == 'mean':
            return self.sum_scores / self.total
        elif self.reduction == 'sum':
            return self.sum_scores


    @property
    def is_differentiable(self) -> bool:
        return True
