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
import pickle

import numpy as np
import pytest
import torch
from scipy.linalg import sqrtm as scipy_sqrtm
from torch.utils.data import Dataset

from torchmetrics.image_quality.fid import FID, sqrtm, _update_cov, _update_mean
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE
from tests.helpers.datasets import TrialMNIST

torch.manual_seed(42)


def test_update_functions(tmpdir):
    """ Test that updating the estimates are equal to estimating them on all data """
    data = torch.randn(100, 2)
    batches = data.chunk(4)

    def _mean_cov(data):
        data = data.numpy()
        return np.mean(data, axis=0), np.cov(data, rowvar=False)

    mean_update, cov_update, size_update = torch.zeros(2), torch.zeros(2, 2), torch.zeros(1)
    for batch in batches:
        new_mean = _update_mean(mean_update, size_update, batch)
        new_cov = _update_cov(cov_update, mean_update, new_mean, batch)
        assert not torch.allclose(new_mean, mean_update), "mean estimate did not update"
        assert not torch.allclose(new_cov, cov_update), "covariance estimate did not update"

        size_update += batch.shape[0]
        mean_update = new_mean
        cov_update = new_cov

    mean, cov = _mean_cov(data)

    assert torch.allclose(torch.tensor(mean).float(), mean_update), \
        "updated mean does not correspond to mean of all data"
    assert torch.allclose(torch.tensor(cov).float(), cov_update / (size_update-1)), \
        "updated covariance does not correspond to covariance of all data"


@pytest.mark.parametrize("matrix_size", [2, 10, 100, 500])
def test_matrix_sqrt(matrix_size):
    """ test that metrix sqrt function works as expected """

    def generate_cov(n):
        data = torch.randn(2 * n, n)
        return (data - data.mean(dim=0)).T @ (data - data.mean(dim=0))

    cov1 = generate_cov(matrix_size)
    cov2 = generate_cov(matrix_size)

    scipy_res = scipy_sqrtm((cov1 @ cov2).numpy()).real
    tm_res = sqrtm(cov1 @ cov2)

    assert torch.allclose(torch.tensor(scipy_res), tm_res, atol=1e-2)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason='test requires torch-fidelity')
def test_fid_pickle():
    """ Assert that we can initialize the metric and pickle it"""
    metric = FID()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason='test requires torch-fidelity')
def test_fid_same_input():
    """ if real and fake are update on the same data the fid score should be 0 """
    metric = FID(feature=192)

    for _ in range(2):
        img = torch.randint(0, 255, (5, 3, 299, 299), dtype=torch.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(metric.real_mean, metric.fake_mean)
    assert torch.allclose(metric.real_cov, metric.fake_cov)
    assert torch.allclose(metric.real_nobs, metric.fake_nobs)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        
    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.imgs.shape[0]

@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count()>=1),
                    reason='test is too slow without gpu')
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason='test requires torch-fidelity')
def test_compare_fid(tmpdir, feature = 2048):
    """ check that the hole pipeline give the same result as torch-fidelity """
    from torch_fidelity import calculate_metrics

    metric = FID(feature=feature).cuda().double()  
    
    img1 = TrialMNIST(tmpdir, num_samples=1500, digits = (0, 1, 2)).data.unsqueeze(1).repeat(1,3,1,1)
    img2 = TrialMNIST(tmpdir, num_samples=1500, digits = (3, 4, 5)).data.unsqueeze(1).repeat(1,3,1,1)
    import pdb
    pdb.set_trace()
    batch_size = 100
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size*i:batch_size*(i+1)].cuda(), real=True)
        
    for i in range(img2.shape[0] // batch_size):
        metric.update(img2[batch_size*i:batch_size*(i+1)].cuda(), real=False)

    tm_res = metric.compute()

    torch_fid = calculate_metrics(
        _ImgDataset(img1),
        _ImgDataset(img2),
        fid=True, feature_layer_fid=str(feature)
    )

    assert torch.allclose(tm_res.cpu(), torch.tensor([torch_fid['frechet_inception_distance']]))
