from numpy.lib.type_check import real
from copy import deepcopy
from typing import Any, List, Optional, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

# Implementation functions inspired by https://github.com/jybai/generative-memorization-benchmark/blob/main/src/competition_scoring.py
def _normalize_rows(x: Tensor):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A PyTorch tensor of shape (n, m)

    Returns:
     ``x``: The normalized (by row) PyTorch tensor.
    """
    return x / torch.norm(x, dim=1, keepdim=True)

def _distance_thresholding(d: Tensor, eps=0.1):
    if d < eps:
        return d
    else:
        return 1

def _compute_cosine_distance(features1: Tensor, features2: Tensor):
    features1_nozero = features1[torch.sum(features1, dim=1) != 0]
    features2_nozero = features2[torch.sum(features2, dim=1) != 0]
    norm_f1 = _normalize_rows(features1_nozero)
    norm_f2 = _normalize_rows(features2_nozero)

    d = 1.0 - torch.abs(torch.matmul(norm_f1, norm_f2.t()))
    mean_min_d = torch.mean(torch.min(d, dim=1).values)
    return mean_min_d

def _mifid_compute(mu1: Tensor, sigma1: Tensor, features1: Tensor, mu2: Tensor, sigma2: Tensor, features2: Tensor):
    fid_value = _compute_fid(mu1, sigma1, mu2, sigma2)
    distance = _compute_cosine_distance(features1, features2)
    distance_thr = _distance_thresholding(distance, eps=0.1)
    mifid = fid_value / (distance_thr + 10e-15)
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
            num_features = feature
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
            num_features = self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.add_state("real_features_stacked", torch.zeros((0, num_features)).double(), dist_reduce_fx="cat")
        self.add_state("fake_features_stacked", torch.zeros((0, num_features)).double(), dist_reduce_fx="cat")

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_stacked = torch.cat((self.real_features_stacked, features), dim=0)
        else:
            self.fake_features_stacked = torch.cat((self.fake_features_stacked, features), dim=0)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""

        mean_real = torch.mean(self.real_features_stacked, dim=0).unsqueeze(0)
        mean_fake = torch.mean(self.fake_features_stacked, dim=0).unsqueeze(0)

        cov_real = torch.cov(self.real_features_stacked.t())
        cov_fake = torch.cov(self.fake_features_stacked.t())

        return _mifid_compute(
            mean_real.squeeze(0),
            cov_real,
            self.real_features_stacked,
            mean_fake.squeeze(0),
            cov_fake,
            self.fake_features_stacked,
        ).to(self.orig_dtype)

    def to(self, device):
        self.inception = self.inception.to(device)
        return super().to(device)

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_stacked = deepcopy(self.real_features_stacked)
            super().reset()
            self.real_features_stacked = real_features_stacked
        else:
            super().reset()
