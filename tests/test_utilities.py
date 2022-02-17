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
from torch import tensor

from torchmetrics.utilities import rank_zero_debug, rank_zero_info, rank_zero_warn
from torchmetrics.utilities.data import _flatten, _flatten_dict, get_num_classes, to_categorical, to_onehot
from torchmetrics.utilities.distributed import class_reduce, reduce


def test_prints():
    rank_zero_debug("DEBUG")
    rank_zero_info("INFO")
    rank_zero_warn("WARN")


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


def test_onehot():
    test_tensor = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    )

    assert test_tensor.shape == (2, 5)
    assert expected.shape == (2, 10, 5)

    onehot_classes = to_onehot(test_tensor, num_classes=10)
    onehot_no_classes = to_onehot(test_tensor)

    assert torch.allclose(onehot_classes, onehot_no_classes)

    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape

    assert torch.allclose(expected.to(onehot_no_classes), onehot_no_classes)
    assert torch.allclose(expected.to(onehot_classes), onehot_classes)


def test_to_categorical():
    test_tensor = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    ).to(torch.float)

    expected = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected.to(result.dtype))


@pytest.mark.parametrize(
    ["preds", "target", "num_classes", "expected_num_classes"],
    [
        (torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), 10, 10),
        (torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
        (torch.rand(32, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
    ],
)
def test_get_num_classes(preds, target, num_classes, expected_num_classes):
    assert get_num_classes(preds, target, num_classes) == expected_num_classes


def test_flatten_list():
    """Check that _flatten utility function works as expected."""
    inp = [[1, 2, 3], [4, 5], [6]]
    out = _flatten(inp)
    assert out == [1, 2, 3, 4, 5, 6]


def test_flatten_dict():
    """Check that _flatten_dict utility function works as expected."""
    inp = {"a": {"b": 1, "c": 2}, "d": 3}
    out = _flatten_dict(inp)
    assert out == {"b": 1, "c": 2, "d": 3}
