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
import pytest
import pickle
import torch
from scipy.linalg import sqrtm
from torchmetrics.image_generation.fid import FID, _update_cov, _update_mean, _matrix_sqrt
from torchmetrics.utilities import _TORCHVISION_AVAILABLE
#from piq import piq_FID

torch.manual_seed(42)

def test_update_functions(tmpdir):
    """ Test that updating the estimates are equal to estimating them on all data """
    data = torch.randn(100, 2)
    batch1, batch2 = data.chunk(2)
    
    def _mean_cov(data):
        mean = data.mean(0)
        diff = data - mean
        cov = diff.T @ diff
        return mean, cov
    
    mean_update, cov_update, size_update = torch.zeros(2), torch.zeros(2,2), torch.zeros(1)
    for batch in [batch1, batch2]:
        new_mean = _update_mean(mean_update, size_update, batch) 
        new_cov = _update_cov(cov_update, mean_update, new_mean, batch)
        
        assert not torch.allclose(new_mean, mean_update), "mean estimate did not update"
        assert not torch.allclose(new_cov, cov_update), "covariance estimate did not update"
        
        size_update += batch.shape[0]
        mean_update = new_mean
        cov_update = new_cov
        
    mean, cov = _mean_cov(data)
    
    assert torch.allclose(mean, mean_update), "updated mean does not correspond to mean of all data"
    assert torch.allclose(cov, cov_update), "updated covariance does not correspond to covariance of all data"

@pytest.mark.parametrize("matrix_size", [2, 10, 100, 2048])
def test_matrix_sqrt(matrix_size):
    """ test that metrix sqrt function works as expected """    
    matrix = torch.randn(matrix_size, matrix_size)
    scipy_res = sqrtm(matrix.numpy()).real
    tm_res, _ = _matrix_sqrt(matrix)
    import pdb
    pdb.set_trace()
    assert torch.allclose(torch.tensor(scipy_res), tm_res)

@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason='test requires torchvision')
def test_fid_pickle():
    """ Assert that we can initialize the metric and pickle it"""
    metric = FID()
    assert metric
    
    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason='test requires torchvision')
def test_fid_same_input():
    """ # if real and fake are update on the same data the fid score should be 0 """
    metric = FID()
    
    
    for _ in range(2):
        img = torch.rand(5, 3, 299, 299)
        metric.update(img, real=True)
        metric.update(img, fake=True)
        
    assert torch.allclose(metric.mean_real, metric.mean_fake)
    assert torch.allclose(metric.cov_real, metric.cov_fake)
    assert torch.allclose(metric.nobs_real, metric.nobs_fake)
    
    val = metric.compute()
    assert val == 0
        