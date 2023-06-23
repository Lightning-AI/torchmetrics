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
from copy import deepcopy
from typing import Any, List, Optional, Union

import torch
from numpy.lib.type_check import real
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def _compute_cosine_distance(features1: Tensor, features2: Tensor, eps: float = 0.1):
    features1_nozero = features1[torch.sum(features1, dim=1) != 0]
    features2_nozero = features2[torch.sum(features2, dim=1) != 0]

    # normalize
    norm_f1 = features1_nozero / torch.norm(features1_nozero, dim=1, keepdim=True)
    norm_f2 = features2_nozero / torch.norm(features2_nozero, dim=1, keepdim=True)

    d = 1.0 - torch.abs(torch.matmul(norm_f1, norm_f2.t()))
    mean_min_d = torch.mean(d.min(dim=1).values)
    mean_min_d = mean_min_d if mean_min_d > eps else eps * torch.ones_like(mean_min_d)
    return mean_min_d


def _mifid_compute(mu1: Tensor, sigma1: Tensor, features1: Tensor, mu2: Tensor, sigma2: Tensor, features2: Tensor):
    fid_value = _compute_fid(mu1, sigma1, mu2, sigma2)
    distance = _compute_cosine_distance(features1, features2)
    mifid = fid_value / (distance + 10e-15)
    return mifid


class MemorizationInformedFrechetInceptionDistance(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_stacked: Tensor
    fake_features_stacked: Tensor

    def __init__(
        self,
        feature: Union[int, Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(feature, int):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "MemorizationInformedFrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = [64, 192, 768, 2048]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])

        elif isinstance(feature, Module):
            self.inception = feature
            dummy_image = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8, device=self.inception.device)
            self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        mean_real = torch.mean(real_features, dim=0).unsqueeze(0)
        mean_fake = torch.mean(fake_features, dim=0).unsqueeze(0)

        cov_real = torch.cov(real_features.t())
        cov_fake = torch.cov(fake_features.t())

        return _mifid_compute(
            mean_real.squeeze(0),
            cov_real,
            real_features,
            mean_fake.squeeze(0),
            cov_fake,
            fake_features,
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()
