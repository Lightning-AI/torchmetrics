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
import pytest
import torch
from torchmetrics.functional.image.perceptual_path_length import perceptual_path_length
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


class DummyGenerator(torch.nn.Module):
    """From https://github.com/toshas/torch-fidelity/blob/master/examples/sngan_cifar10.py."""

    def __init__(self, z_size) -> None:
        super().__init__()
        self.z_size = z_size
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=(1, 1)),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        """Generate images from latent vectors."""
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = 255 * (fake.clamp(-1, 1) * 0.5 + 0.5)
            fake = fake.to(torch.uint8)
        return fake

    def sample(self, num_samples):
        """Sample latent vectors."""
        return torch.randn(num_samples, self.z_size)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch_fidelity")
@pytest.mark.parametrize(
    ("argument", "match"),
    [
        ({"num_samples": 0}, "Argument `num_samples` must be a positive integer, but got 0."),
        ({"conditional": 2}, "Argument `conditional` must be a boolean, but got 2."),
    ],
)
def test_raises_error_on_wrong_arguments(argument, match):
    """Test that appropriate errors are raised on wrong arguments."""
    with pytest.raises(ValueError, match=match):
        perceptual_path_length(DummyGenerator(128), **argument)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch_fidelity")
def test_compare():
    """Test something."""
    import torch_fidelity

    generator = DummyGenerator(128)

    compare = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(generator, 128, "normal", 10),
        input1_model_num_samples=10000,
        ppl=True,
        ppl_reduction="none",
    )

    result = perceptual_path_length(generator)
    assert result == compare["ppl"]
