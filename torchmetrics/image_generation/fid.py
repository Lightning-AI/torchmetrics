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
from functools import wraps
from typing import Any, Callable, Optional

import torch
from torch import nn

from torchmetrics.metric import Metric
from torchmetrics.utilities import _TORCHVISION_AVAILABLE, MisconfigurationException

if _TORCHVISION_AVAILABLE:
    from torchvision import models
    from torchvision import transforms


class _Identity(nn.Module):
    """ Module that does nothing. Use to overwrite layers to be no-op layers """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _notrain(func):
    """ Used wrap the `.train` method of any model, such that it will always stay in evaluation mode """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(False)

    return wrapper


# def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
#     """ Calculates the matrix square root of a single 2D matrix of size [N,N] (needs to be square) """
#     eigval, eigvec = torch.eig(matrix, eigenvectors=True)
#     eigval = torch.view_as_complex(eigval.contiguous())
#     eigval_sqrt = eigval.sqrt()
#     eigval_sqrt = torch.view_as_real(eigval_sqrt)[:, 0]
#     return eigvec @ torch.diag(eigval_sqrt) @ torch.inverse(eigvec)


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error

#https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138/5
def _matrix_sqrt(matrix: torch.Tensor, num_iters: int = 100):
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {matrix.dim()}, expected {expected_num_dims}')

    if num_iters <= 0:
        raise ValueError(f'Number of iteration equals {num_iters}, expected greater than 0')

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.]).to(error), atol=1e-5):
            break
        
        import pdb
        pdb.set_trace()

    return s_matrix, error


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
    r"""
    Calculates `Fréchet inception distance (FID) <https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance>_`
    which is used to access the quality of generated images. Given by

    .. math::
        FID = |\mu - \mu_w| + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3 [1]
    features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the multivariate normal
    distribution estimated from Inception v3 features calculated on generated (fake) images. The metric was
    originally proposed in [1].
    
    The input is expected to be mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are 
    expected to be at least 299. The boolian flag ``real`` determines if the images should update the statistics
    of the real distribution or the fake distribution.

    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.

    .. note:: it is recommend to use atleast a few thousand samples to calculate fid


    Args:
        perform_nomalization: if ``True`` will normalize the input by mean [0.485, 0.456, 0.406] 
            and std = [0.229, 0.224, 0.225] ()
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    [1] Rethinking the Inception Architecture for Computer Vision
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    https://arxiv.org/abs/1512.00567

    [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
    Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
    https://arxiv.org/abs/1706.08500
    """
    def __init__(
        self,
        images_or_features: str = 'images',
        feature_size: Optional[int] = 2048,
        perform_normalization: bool = True,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        allowed_input = ('images', 'features')
        if images_or_features not in allowed_input:
            raise MisconfigurationException("Expected argument `images_or_features` to be one of the following"
                                            " {allowed")
        self.images_or_features = images_or_features
        
        if self.images_or_features == 'images' and not _TORCHVISION_AVAILABLE:
            raise MisconfigurationException("FID metric requires torchvision to be installed for downloading"
                                            " inception v3 network")
            
        if self.images_or_features == 'images':
            if perform_normalization:
                self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.perform_normalization = perform_normalization
            self.inception = models.inception_v3(pretrained=True)
            # remove the classification layer
            self.inception.fc = _Identity()
            # disable going into training mode
            self.inception.eval()
            self.inception.train = _notrain(self.inception.train)           
            feature_size = 2048
            
        self.add_state("real_mean", torch.zeros(feature_size), dist_reduce_fx="mean")
        self.add_state("real_cov", torch.zeros(feature_size, feature_size), dist_reduce_fx="mean")
        self.add_state("real_nobs", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fake_mean", torch.zeros(feature_size, feature_size), dist_reduce_fx="mean")
        self.add_state("fake_cov", torch.zeros(feature_size, feature_size), dist_reduce_fx="mean")
        self.add_state("fake_nobs", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, imgs_or_feature: torch.Tensor, real: bool = True) -> None:
        """ Update distributions with new data
        Args:
            imgs: image data of shape ``[N, 3, 299, 299]``
            real: boolian indicating if the images comes from the true data distribution or the fake data
                distribution (generated images)
        """
        self._check_valid_input(imgs_or_feature)
        
        if self.images_or_features == 'images':
            if self.perform_normalization:
                imgs = self.normalizer(imgs_or_feature)
            incep_score = self.inception(imgs)
        else:
            incep_score = imgs_or_feature
        
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

    def compute(self) -> torch.Tensor:
        """ Returns the FID score """
        mean_diff = torch.abs(self.real_mean - self.fake_mean)
        cov_diff = torch.trace(self.real_cov + self.fake_cov - 2 * _matrix_sqrt(self.real_cov @ self.fake_cov))
        return (mean_diff * mean_diff).sum() + cov_diff
    
    def _check_valid_input(self, img_or_feature):
        if self.images_or_features == 'images':
            if img_or_feature.ndim != 4:
                raise ValueError('Input should be a [B,C,H,W] image tensor (B=batch size, C=number of channels,'
                                 ' H=height of images,W=weidth of images')
            img_shape = img_or_feature.shape
            if img_shape[-1] < 299 or img_shape[-2] < 299 or img_shape[-3] != 3:
                raise ValueError('Input tensor is expected to have 3 color channels and both the width and hight'
                                 ' dimension being atleast 299.')
    
            if self.perform_normalization and 0 > img_or_feature.min() and img_or_feature.max() > 1:
                raise ValueError('Expected all values of the input tensor to be in the [0,1] range but found'
                                 ' minimum value {img.min()} and maximum value {img.max()}')
        else:
            if img_or_feature.ndim != 2:
                raise ValueError('input should be a [B,D] feature tensor (B=batch size, D=feature size)')
