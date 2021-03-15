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
from typing import Any, Callable, Optional

import torch
from torch import nn
from pytorch_lightning.metrics import Metric
import torchvision
from functools import wraps

class _Identity(nn.Module):
    """ Module that does nothing. Use to overwrite layers to be no-op layers """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def notrain(func):
    """ Used wrap the `.train` method of any model, such that it will always stay in evaluation mode """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(False)
    return wrapper
        
def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """ Calculates the matrix square root of a single 2D matrix of size [N,N] (needs to be square) """
    eigval, eigvec = torch.eig(matrix, eigenvectors=True)
    eigval = torch.where(eigval > 0, eigval.sqrt(), torch.zeros_like(eigval))
    return eigvec @ eigval[:,0].diag_embed() @ torch.inverse(eigvec)

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
    """ 
    Calculates `Fréchet inception distance (FID) <https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance>_`
    which is used to access the quality of generated images. Given by
    
    .. math:: 
        FID = |\mu - \mu_w| + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})
    
    where :math: \mathcal{N}(\mu, \Sigma) is the multivariate normal distribution estimated from Inception v3 [1]
    features calculated on real life images and :math: \mathcal{N}(\mu_w, \Sigma_w) is the multivariate normal
    distribution estimated from Inception v3 features calculated on generated (fake) images. The metric was
    originally proposed in [1]. 
    
    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.
    
    .. note:: it is recommend to use atleast a few thousand samples to calculate fid 
    
    [1] Rethinking the Inception Architecture for Computer Vision 
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    https://arxiv.org/abs/1512.00567
    
    [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
    Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
    https://arxiv.org/abs/1706.08500
    
    """
    def __init__(self,                 
        compute_on_step: bool = True,
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
        self.inception = torchvision.models.inception_v3(pretrained=True)
        # remove the classification layer
        self.inception.fc = _Identity()
        # disable going into training mode
        self.inception.eval()
        self.inception.train = notrain(self.inception.train)
        
        # we need to deal with syncronization in specialized way, so we turn it of 
        self.add_state("real_mean", torch.zeros(2048), dist_reduce_fx=None)
        self.add_state("real_cov", torch.zeros(2048, 2048), dist_reduce_fx=None)
        self.add_state("real_nobs", torch.zeros(1), dist_reduce_fx = None)
        self.add_state("fake_mean", torch.zeros(2048, 2048), dist_reduce_fx=None)
        self.add_state("fake_cov", torch.zeros(2048, 2048), dist_reduce_fx=None)
        self.add_state("fake_nobs", torch.zeros(1), dist_reduce_fx = None)
        
    def update(self, imgs: torch.Tensor, real: bool = True) -> None:
        """ Update distributions with new data
        Args:
            imgs: image data of shape ``[N, 3, 299, 299]``
            real: boolian indicating if the images comes from the true data distribution or the fake data
                distribution (generated images)
        """
        incep_score = self.inception(imgs)
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
        cov_diff = torch.trace(self.real_cov + self.fake_cov - 2*_matrix_sqrt(self.real_cov @ self.fake_cov))
        return (mean_diff * mean_diff).sum() + cov_diff

data = torch.randn(50, 3, 299, 299)
metric = FID()
for d in [data[5*i:5*(i+1)] for i in range(10)]:
    metric.update(d, real=True)

incep_scores = metric.inception(data)
mean = incep_scores.mean(dim=0)
cov = (incep_scores - mean).T @ (incep_scores - mean)
