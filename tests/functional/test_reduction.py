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

from torchmetrics.utilities.distributed import class_reduce, reduce


def test_reduce():
    start_tensor = torch.rand(50, 40, 30)

    assert torch.allclose(reduce(start_tensor, "elementwise_mean"), torch.mean(start_tensor))
    assert torch.allclose(reduce(start_tensor, "sum"), torch.sum(start_tensor))
    assert torch.allclose(reduce(start_tensor, "none"), start_tensor)

    with pytest.raises(ValueError):
        reduce(start_tensor, "error_reduction")


def test_class_reduce():
    num = torch.randint(1, 10, (100,)).float()
    denom = torch.randint(10, 20, (100,)).float()
    weights = torch.randint(1, 100, (100,)).float()

    assert torch.allclose(class_reduce(num, denom, weights, "micro"), torch.sum(num) / torch.sum(denom))
    assert torch.allclose(class_reduce(num, denom, weights, "macro"), torch.mean(num / denom))
    assert torch.allclose(
        class_reduce(num, denom, weights, "weighted"), torch.sum(num / denom * (weights / torch.sum(weights)))
    )
    assert torch.allclose(class_reduce(num, denom, weights, "none"), num / denom)
