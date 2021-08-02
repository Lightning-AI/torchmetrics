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
import torch
from torch import Tensor

from torchmetrics.functional import image_gradients


def test_invalid_input_img_type():
    """Test Whether the module successfully handles invalid input data type"""
    invalid_dummy_input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    with pytest.raises(TypeError):
        image_gradients(invalid_dummy_input)


def test_invalid_input_ndims():
    """
    Test whether the module successfully handles invalid number of dimensions
    of input tensor
    """

    BATCH_SIZE = 1
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    image = torch.arange(0, BATCH_SIZE * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    image = torch.reshape(image, (HEIGHT, WIDTH))

    with pytest.raises(RuntimeError):
        image_gradients(image)


def test_multi_batch_image_gradients():
    """Test whether the module correctly calculates gradients for known input
    with non-unity batch size.Example input-output pair taken from TF's implementation of i
    mage-gradients
    """

    BATCH_SIZE = 5
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    single_channel_img = torch.arange(0, 1 * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    single_channel_img = torch.reshape(single_channel_img, (CHANNELS, HEIGHT, WIDTH))
    image = torch.stack([single_channel_img for _ in range(BATCH_SIZE)], dim=0)

    true_dy = [
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    true_dy = Tensor(true_dy)

    dy, dx = image_gradients(image)

    for batch_id in range(BATCH_SIZE):
        assert torch.allclose(dy[batch_id, 0, :, :], true_dy)
    assert dy.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)
    assert dx.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)


def test_image_gradients():
    """Test whether the module correctly calculates gradients for known input.
    Example input-output pair taken from TF's implementation of image-gradients
    """

    BATCH_SIZE = 1
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    image = torch.arange(0, BATCH_SIZE * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    image = torch.reshape(image, (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))

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
