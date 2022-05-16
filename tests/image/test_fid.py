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

import pytest
import torch
from scipy.linalg import sqrtm as scipy_sqrtm
from torch.nn import Module
from torch.utils.data import Dataset

from torchmetrics.image.fid import FrechetInceptionDistance, sqrtm
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

torch.manual_seed(42)


@pytest.mark.parametrize("matrix_size", [2, 10, 100, 500])
def test_matrix_sqrt(matrix_size):
    """test that metrix sqrt function works as expected."""

    def generate_cov(n):
        data = torch.randn(2 * n, n)
        return (data - data.mean(dim=0)).T @ (data - data.mean(dim=0))

    cov1 = generate_cov(matrix_size)
    cov2 = generate_cov(matrix_size)

    scipy_res = scipy_sqrtm((cov1 @ cov2).numpy()).real
    tm_res = sqrtm(cov1 @ cov2)
    assert torch.allclose(torch.tensor(scipy_res).float().trace(), tm_res.trace())


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.metric = FrechetInceptionDistance()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.inception.training, "FID metric was changed to training mode which should not happen"


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_fid_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = FrechetInceptionDistance()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_fid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `FrechetInceptionDistance` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        _ = FrechetInceptionDistance()

    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            _ = FrechetInceptionDistance(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="FID metric requires that `Torch-fidelity` is installed."
            " Either install as `pip install torchmetrics[image-quality]` or `pip install torch-fidelity`.",
        ):
            _ = FrechetInceptionDistance()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        _ = FrechetInceptionDistance(feature=[1, 2])


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_fid_same_input(feature):
    """if real and fake are update on the same data the fid score should be
    0."""
    metric = FrechetInceptionDistance(feature=feature)

    for _ in range(2):
        img = torch.randint(0, 255, (10, 3, 299, 299), dtype=torch.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(torch.cat(metric.real_features, dim=0), torch.cat(metric.fake_features, dim=0))

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.imgs.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("equal_size", [False, True])
def test_compare_fid(tmpdir, equal_size, feature=2048):
    """check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = FrechetInceptionDistance(feature=feature).cuda()

    n = 100
    m = 100 if equal_size else 90

    # Generate some synthetic data
    img1 = torch.randint(0, 180, (n, 3, 299, 299), dtype=torch.uint8)
    img2 = torch.randint(100, 255, (m, 3, 299, 299), dtype=torch.uint8)

    batch_size = 10
    for i in range(n // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)

    for i in range(m // batch_size):
        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)

    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1),
        input2=_ImgDataset(img2),
        fid=True,
        feature_layer_fid=str(feature),
        batch_size=batch_size,
        save_cpu_ram=True,
    )

    tm_res = metric.compute()

    assert torch.allclose(tm_res.cpu(), torch.tensor([torch_fid["frechet_inception_distance"]]), atol=1e-3)


@pytest.mark.parametrize("reset_real_features", [True, False])
def test_reset_real_features_arg(reset_real_features):
    metric = FrechetInceptionDistance(feature=64, reset_real_features=reset_real_features)

    metric.update(torch.randint(0, 180, (2, 3, 299, 299), dtype=torch.uint8), real=True)
    metric.update(torch.randint(0, 180, (2, 3, 299, 299), dtype=torch.uint8), real=False)

    assert len(metric.real_features) == 1
    assert list(metric.real_features[0].shape) == [2, 64]

    assert len(metric.fake_features) == 1
    assert list(metric.fake_features[0].shape) == [2, 64]

    metric.reset()

    # fake features should always reset
    assert len(metric.fake_features) == 0

    if reset_real_features:
        assert len(metric.real_features) == 0
    else:
        assert len(metric.real_features) == 1
        assert list(metric.real_features[0].shape) == [2, 64]
