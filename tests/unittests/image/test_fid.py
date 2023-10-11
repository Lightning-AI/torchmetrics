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
import pickle
from contextlib import nullcontext as does_not_raise
from functools import partial

import pytest
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

torch.manual_seed(42)


@pytest.mark.skipif(_TORCH_FIDELITY_AVAILABLE, reason="test only works if torch-fidelity is not installed")
def test_no_train_network_missing_torch_fidelity():
    """Assert that NoTrainInceptionV3 raises an error if torch-fidelity is not installed."""
    with pytest.raises(
        ModuleNotFoundError, match="NoTrainInceptionV3 module requires that `Torch-fidelity` is installed.*"
    ):
        NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"])


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(Module):
        def __init__(self) -> None:
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
    """If real and fake are update on the same data the fid score should be 0."""
    metric = FrechetInceptionDistance(feature=feature)

    for _ in range(2):
        img = torch.randint(0, 255, (10, 3, 299, 299), dtype=torch.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(metric.real_features_sum, metric.fake_features_sum)
    assert torch.allclose(metric.real_features_cov_sum, metric.fake_features_cov_sum)
    assert torch.allclose(metric.real_features_num_samples, metric.fake_features_num_samples)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


class _ImgDataset(Dataset):
    def __init__(self, imgs) -> None:
        self.imgs = imgs

    def __getitem__(self, idx) -> torch.Tensor:
        return self.imgs[idx]

    def __len__(self) -> int:
        return self.imgs.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("equal_size", [False, True])
def test_compare_fid(tmpdir, equal_size, feature=768):
    """Check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = FrechetInceptionDistance(feature=feature).cuda()

    n, m = 100, 100 if equal_size else 90

    # Generate some synthetic data
    torch.manual_seed(42)
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
    """Test that `reset_real_features` argument works as expected."""
    metric = FrechetInceptionDistance(feature=64, reset_real_features=reset_real_features)

    metric.update(torch.randint(0, 180, (2, 3, 299, 299), dtype=torch.uint8), real=True)
    metric.update(torch.randint(0, 180, (2, 3, 299, 299), dtype=torch.uint8), real=False)

    assert metric.real_features_num_samples == 2
    assert metric.real_features_sum.shape == torch.Size([64])
    assert metric.real_features_cov_sum.shape == torch.Size([64, 64])

    assert metric.fake_features_num_samples == 2
    assert metric.fake_features_sum.shape == torch.Size([64])
    assert metric.fake_features_cov_sum.shape == torch.Size([64, 64])

    metric.reset()

    # fake features should always reset
    assert metric.fake_features_num_samples == 0

    if reset_real_features:
        assert metric.real_features_num_samples == 0
    else:
        assert metric.real_features_num_samples == 2
        assert metric.real_features_sum.shape == torch.Size([64])
        assert metric.real_features_cov_sum.shape == torch.Size([64, 64])


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_arg(normalize):
    """Test that normalize argument works as expected."""
    img = torch.rand(2, 3, 299, 299)
    metric = FrechetInceptionDistance(normalize=normalize)

    context = (
        partial(
            pytest.raises, expected_exception=ValueError, match="Expecting image as torch.Tensor with dtype=torch.uint8"
        )
        if not normalize
        else does_not_raise
    )

    with context():
        metric.update(img, real=True)


def test_not_enough_samples():
    """Test that an error is raised if not enough samples were provided."""
    img = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
    metric = FrechetInceptionDistance()
    metric.update(img, real=True)
    metric.update(img, real=False)
    with pytest.raises(
        RuntimeError, match="More than one sample is required for both the real and fake distributed to compute FID"
    ):
        metric.compute()


def test_dtype_transfer_to_submodule():
    """Test that change in dtype also changes the default inception net."""
    imgs = torch.randn(1, 3, 256, 256)
    imgs = ((imgs.clamp(-1, 1) / 2 + 0.5) * 255).to(torch.uint8)

    metric = FrechetInceptionDistance(feature=64)
    metric.set_dtype(torch.float64)

    out = metric.inception(imgs)
    assert out.dtype == torch.float64
