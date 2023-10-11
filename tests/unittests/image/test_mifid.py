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
from contextlib import nullcontext as does_not_raise
from functools import partial

import numpy as np
import pytest
import torch
from scipy.linalg import sqrtm
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance, NoTrainInceptionV3
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def _compare_mifid(preds, target, cosine_distance_eps: float = 0.1):
    """Reference implementation.

    Implementation taken from:
    https://github.com/jybai/generative-memorization-benchmark/blob/main/src/competition_scoring.py

    Adjusted slightly to work with our code. We replace the feature extraction with our own, since we already check in
    FID that we use the correct feature extractor. This saves us from needing to download tensorflow for comparison.

    """

    def normalize_rows(x: np.ndarray):
        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    def cosine_distance(features1, features2):
        features1_nozero = features1[np.sum(features1, axis=1) != 0]
        features2_nozero = features2[np.sum(features2, axis=1) != 0]
        norm_f1 = normalize_rows(features1_nozero)
        norm_f2 = normalize_rows(features2_nozero)

        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
        return np.mean(np.min(d, axis=1))

    def distance_thresholding(d, eps):
        return d if d < eps else 1

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise Exception(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_activation_statistics(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma, act

    def calculate_mifid(m1, s1, features1, m2, s2, features2):
        fid = calculate_frechet_distance(m1, s1, m2, s2)
        distance = cosine_distance(features1, features2)
        return fid, distance

    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(768)])
    preds_act = net(preds).numpy()
    target_act = net(target).numpy()

    m1, s1, features1 = calculate_activation_statistics(preds_act)
    m2, s2, features2 = calculate_activation_statistics(target_act)

    fid_private, distance_private = calculate_mifid(m1, s1, features1, m2, s2, features2)
    distance_private_thresholded = distance_thresholding(distance_private, cosine_distance_eps)
    return fid_private / (distance_private_thresholded + 1e-15)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.metric = MemorizationInformedFrechetInceptionDistance()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.inception.training, "MiFID metric was changed to training mode which should not happen"


def test_mifid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            _ = MemorizationInformedFrechetInceptionDistance(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="FID metric requires that `Torch-fidelity` is installed."
            " Either install as `pip install torchmetrics[image-quality]` or `pip install torch-fidelity`.",
        ):
            _ = MemorizationInformedFrechetInceptionDistance()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        _ = MemorizationInformedFrechetInceptionDistance(feature=[1, 2])

    with pytest.raises(ValueError, match="Argument `cosine_distance_eps` expected to be a float greater than 0"):
        _ = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=-1)

    with pytest.raises(ValueError, match="Argument `cosine_distance_eps` expected to be a float greater than 0"):
        _ = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=1.1)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_fid_same_input(feature):
    """If real and fake are update on the same data the fid score should be 0."""
    metric = MemorizationInformedFrechetInceptionDistance(feature=feature)

    for _ in range(2):
        img = torch.randint(0, 255, (10, 3, 299, 299), dtype=torch.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(torch.cat(metric.real_features, dim=0), torch.cat(metric.fake_features, dim=0))

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("equal_size", [False, True])
def test_compare_mifid(equal_size):
    """Check that our implementation of MIFID is correct by comparing it to the original implementation."""
    metric = MemorizationInformedFrechetInceptionDistance(feature=768).cuda()

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

    compare_val = _compare_mifid(img1, img2)
    tm_res = metric.compute()

    assert torch.allclose(tm_res.cpu(), torch.tensor(compare_val, dtype=tm_res.dtype), atol=1e-3)


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_arg(normalize):
    """Test that normalize argument works as expected."""
    img = torch.rand(2, 3, 299, 299)
    metric = MemorizationInformedFrechetInceptionDistance(normalize=normalize)

    context = (
        partial(
            pytest.raises, expected_exception=ValueError, match="Expecting image as torch.Tensor with dtype=torch.uint8"
        )
        if not normalize
        else does_not_raise
    )

    with context():
        metric.update(img, real=True)
