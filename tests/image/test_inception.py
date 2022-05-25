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
from torch.nn import Module
from torch.utils.data import Dataset

from torchmetrics.image.inception import InceptionScore
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

torch.manual_seed(42)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.metric = InceptionScore()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert (
        not model.metric.inception.training
    ), "InceptionScore metric was changed to training mode which should not happen"


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_is_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = InceptionScore()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_is_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `InceptionScore` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        InceptionScore()

    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            _ = InceptionScore(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="InceptionScore metric requires that `Torch-fidelity` is installed."
            " Either install as `pip install torchmetrics[image-quality]` or `pip install torch-fidelity`.",
        ):
            InceptionScore()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        InceptionScore(feature=[1, 2])


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_is_update_compute():
    metric = InceptionScore()

    for _ in range(2):
        img = torch.randint(0, 255, (10, 3, 299, 299), dtype=torch.uint8)
        metric.update(img)

    mean, std = metric.compute()
    assert mean >= 0.0
    assert std >= 0.0


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.imgs.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("compute_on_cpu", [True, False])
def test_compare_is(tmpdir, compute_on_cpu):
    """check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = InceptionScore(splits=1, compute_on_cpu=compute_on_cpu).cuda()

    # Generate some synthetic data
    img1 = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8)

    batch_size = 10
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda())

    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1), isc=True, isc_splits=1, batch_size=batch_size, save_cpu_ram=True
    )

    tm_mean, _ = metric.compute()

    assert torch.allclose(tm_mean.cpu(), torch.tensor([torch_fid["inception_score_mean"]]), atol=1e-3)
