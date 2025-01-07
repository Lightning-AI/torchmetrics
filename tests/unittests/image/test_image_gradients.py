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
from torch import Tensor

from torchmetrics.functional import image_gradients


def test_invalid_input_img_type():
    """Test Whether the module successfully handles invalid input data type."""
    invalid_dummy_input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    with pytest.raises(TypeError):
        image_gradients(invalid_dummy_input)


def test_invalid_input_ndims(batch_size=1, height=5, width=5, channels=1):
    """Test whether the module successfully handles invalid number of dimensions of input tensor."""
    image = torch.arange(0, batch_size * height * width * channels, dtype=torch.float32)
    image = torch.reshape(image, (height, width))

    with pytest.raises(RuntimeError):
        image_gradients(image)


def test_multi_batch_image_gradients(batch_size=5, height=5, width=5, channels=1):
    """Test whether the module correctly calculates gradients for known input with non-unity batch size."""
    single_channel_img = torch.arange(0, 1 * height * width * channels, dtype=torch.float32)
    single_channel_img = torch.reshape(single_channel_img, (channels, height, width))
    image = torch.stack([single_channel_img for _ in range(batch_size)], dim=0)

    true_dy = [
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    true_dy = Tensor(true_dy)

    dy, dx = image_gradients(image)

    for batch_id in range(batch_size):
        assert torch.allclose(dy[batch_id, 0, :, :], true_dy)
    assert dy.shape == (batch_size, 1, height, width)
    assert dx.shape == (batch_size, 1, height, width)


def test_image_gradients(batch_size=1, height=5, width=5, channels=1):
    """Test whether the module correctly calculates gradients for known input.

    Example input-output pair taken from TF's implementation of image- gradients

    """
    image = torch.arange(0, batch_size * height * width * channels, dtype=torch.float32)
    image = torch.reshape(image, (batch_size, channels, height, width))

    true_dy = [
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    true_dx = [
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ]

    true_dy = Tensor(true_dy)
    true_dx = Tensor(true_dx)

    dy, dx = image_gradients(image)

    assert torch.allclose(dy, true_dy), "dy fails test"
    assert torch.allclose(dx, true_dx), "dx fails tests"
