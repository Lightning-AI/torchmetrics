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
from typing import Tuple

import torch
from torch import Tensor

from torch_fidility.feature_extractor import FeatureExtractorInceptionV3

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    """
    Credit to: https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py
    """
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Credit to: https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor,
                 eps=1e-6) -> torch.Tensor:
    r"""
    Credit to: https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py
    
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.mm(sigma2))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def _update_mean(old_mean: torch.Tensor, old_nobs: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """ Update a mean estimate given new data
    Args:
        old_mean: current mean estimate
        old_nobs: number of observation until now
        data: data used for updating the estimate
    Returns:
        new_mean: updated mean estimate
    """
    data_size = data.shape[0]
    return (old_mean * old_nobs + data.mean(dim=0) * data_size) / (old_nobs + data_size)


def _update_cov(old_cov: torch.Tensor, old_mean: torch.Tensor, new_mean: torch.Tensor, data: torch.Tensor):
    """ Update a covariance estimate given new data
    Args:
        old_cov: current covariance estimate
        old_mean: current mean estimate
        new_mean: updated mean estimate
        data: data used for updating the estimate
    Returns:
        new_mean: updated covariance estimate
    """
    return old_cov + (data - new_mean).T @ (data - old_mean)


class FID(Metric):
    def __init__(self, feature='2048'):
        if not _TORCH_FIDELITY_AVAILABLE:
            raise ValueError('FID metric requires that Torch-fidelity is installed.'
                             'Either install as `pip install torchmetrics[image-quality]`'
                             ' or `pip install torch-fidelity`')
        if feature not in FeatureExtractorInceptionV3.get_provided_features_list():
            raise ValueError('feature not in list')
            
        self.inception = FeatureExtractorInceptionV3(name='inception-v3-compat',
                                                     feature=[feature])
        
        self.add_state("real_mean", torch.zeros(self.feature_size), dist_reduce_fx="mean")
        self.add_state("real_cov", torch.zeros(self.feature_size, self.feature_size), dist_reduce_fx="mean")
        self.add_state("real_nobs", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fake_mean", torch.zeros(self.feature_size), dist_reduce_fx="mean")
        self.add_state("fake_cov", torch.zeros(self.feature_size, self.feature_size), dist_reduce_fx="mean")
        self.add_state("fake_nobs", torch.zeros(1), dist_reduce_fx="sum")
        
    def update(self, imgs: Tensor, real: bool):
        incep_score = self.inception(imgs)[0]
        
        if real:
            new_mean = _update_mean(self.real_mean, self.real_nobs, incep_score)
            new_cov = _update_cov(self.real_cov, self.real_mean, new_mean, incep_score)
            self.real_mean = new_mean
            self.real_cov = new_cov
            self.real_nobs += incep_score.shape[0]
        else:
            new_mean = _update_mean(self.fake_mean, self.fake_nobs, incep_score)
            new_cov = _update_cov(self.fake_cov, self.fake_mean, new_mean, incep_score)
            self.fake_mean = new_mean
            self.fake_cov = new_cov
            self.fake_nobs += incep_score.shape[0]
        
    def compute(self):
        return _compute_fid()
        
            
            
    