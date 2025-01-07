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
import sys

import numpy as np
import pytest
import torch
from lightning_utilities.test.warning import no_warning_call
from torch import tensor

from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef
from torchmetrics.utilities import check_forward_full_state_property, rank_zero_debug, rank_zero_info, rank_zero_warn
from torchmetrics.utilities.checks import _allclose_recursive
from torchmetrics.utilities.data import (
    _bincount,
    _cumsum,
    _flatten,
    _flatten_dict,
    select_topk,
    to_categorical,
    to_onehot,
)
from torchmetrics.utilities.distributed import class_reduce, reduce
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_2, _TORCH_LESS_THAN_2_6


def test_prints():
    """Test that the different rank zero only functions works as expected."""
    rank_zero_debug("DEBUG")
    rank_zero_info("INFO")
    rank_zero_warn("WARN")


def test_reduce():
    """Test that reduction function works as expected and also raises error on wrong input."""
    start_tensor = torch.rand(50, 40, 30)

    assert torch.allclose(reduce(start_tensor, "elementwise_mean"), torch.mean(start_tensor))
    assert torch.allclose(reduce(start_tensor, "sum"), torch.sum(start_tensor))
    assert torch.allclose(reduce(start_tensor, "none"), start_tensor)

    with pytest.raises(ValueError, match="Reduction parameter unknown."):
        reduce(start_tensor, "error_reduction")


def test_class_reduce():
    """Test that class reduce function works as expected."""
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
    """Test that casting to onehot works as expected."""
    test_tensor = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.stack([
        torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
        torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
    ])

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
    """Test that casting to categorical works as expected."""
    test_tensor = torch.stack([
        torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
        torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
    ]).to(torch.float)

    expected = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected.to(result.dtype))


def test_flatten_list():
    """Check that _flatten utility function works as expected."""
    inp = [[1, 2, 3], [4, 5], [6]]
    out = _flatten(inp)
    assert out == [1, 2, 3, 4, 5, 6]


def test_flatten_dict():
    """Check that _flatten_dict utility function works as expected."""
    inp = {"a": {"b": 1, "c": 2}, "d": 3}
    out_dict, out_dup = _flatten_dict(inp)
    assert out_dict == {"b": 1, "c": 2, "d": 3}
    assert out_dup is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu")
def test_bincount(use_deterministic_algorithms):
    """Test that bincount works in deterministic setting on GPU."""
    x = torch.randint(10, size=(100,))
    # uses custom implementation
    res1 = _bincount(x, minlength=10)

    # uses torch.bincount
    res2 = _bincount(x, minlength=10)

    # explicit call to make sure, that res2 is not by accident using our manual implementation
    res3 = torch.bincount(x, minlength=10)

    # check for correctness
    assert torch.allclose(res1, res2)
    assert torch.allclose(res1, res3)


@pytest.mark.parametrize(("metric_class", "expected"), [(MeanSquaredError, False), (PearsonCorrCoef, True)])
def test_check_full_state_update_fn(capsys, metric_class, expected):
    """Test that the check function works as it should."""
    check_forward_full_state_property(
        metric_class=metric_class,
        input_args={"preds": torch.randn(1000), "target": torch.randn(1000)},
        num_update_to_compare=[10000],
        reps=5,
    )
    captured = capsys.readouterr()
    assert f"Recommended setting `full_state_update={expected}`" in captured.out


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ((torch.ones(2), torch.ones(2)), True),
        ((torch.rand(2), torch.rand(2)), False),
        (([torch.ones(2) for _ in range(2)], [torch.ones(2) for _ in range(2)]), True),
        (([torch.rand(2) for _ in range(2)], [torch.rand(2) for _ in range(2)]), False),
        (({f"{i}": torch.ones(2) for i in range(2)}, {f"{i}": torch.ones(2) for i in range(2)}), True),
        (({f"{i}": torch.rand(2) for i in range(2)}, {f"{i}": torch.rand(2) for i in range(2)}), False),
    ],
)
def test_recursive_allclose(inputs, expected):
    """Test the recursive allclose works as expected."""
    res = _allclose_recursive(*inputs)
    assert res == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
@pytest.mark.xfail(
    sys.platform == "win32" or not _TORCH_LESS_THAN_2_6, reason="test will only fail on non-windows systems"
)
def test_cumsum_still_not_supported(use_deterministic_algorithms):
    """Make sure that cumsum on GPU and deterministic mode still fails.

    If this test begins to pass, it means newer Pytorch versions support this and we can drop internal support.

    """
    with pytest.raises(RuntimeError, match="cumsum_cuda_kernel does not have a deterministic implementation.*"):
        torch.arange(10).float().cuda().cumsum(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_custom_cumsum(use_deterministic_algorithms):
    """Test custom cumsum implementation."""
    # check that cumsum works as expected on non-default cuda device
    device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    x = torch.arange(100).float().to(device)
    with (
        pytest.warns(
            TorchMetricsUserWarning, match="You are trying to use a metric in deterministic mode on GPU that.*"
        )
        if sys.platform != "win32" and _TORCH_LESS_THAN_2_6 and torch.are_deterministic_algorithms_enabled()
        else no_warning_call()
    ):
        ours = _cumsum(x, dim=0)
    ref_np = np.cumsum(x.cpu(), axis=0)
    assert torch.allclose(ours.cpu(), ref_np)


def _reference_topk(x, dim, k):
    x = x.cpu().numpy()
    one_hot = np.zeros((x.shape[0], x.shape[1]), dtype=int)
    if dim == 1:
        for i in range(x.shape[0]):
            one_hot[i, np.argsort(x[i, :], kind="stable")[::-1][:k]] = 1
        return one_hot
    for i in range(x.shape[1]):
        one_hot[np.argsort(x[:, i], kind="stable")[::-1][:k], i] = 1
    return one_hot


@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("dim", [0, 1])
def test_custom_topk(dtype, k, dim):
    """Test custom topk implementation."""
    x = torch.randn(100, 10, dtype=dtype)
    top_k = select_topk(x, dim=dim, topk=k)
    assert top_k.shape == (100, 10)
    assert top_k.dtype == torch.int
    ref = _reference_topk(x, dim=dim, k=k)
    assert torch.allclose(top_k, torch.from_numpy(ref).to(torch.int))


@pytest.mark.skipif(_TORCH_GREATER_EQUAL_2_2, reason="Top-k does not support cpu + half precision")
def test_half_precision_top_k_cpu_raises_error():
    """Test that half precision topk raises error on cpu.

    If this begins to fail, it means newer Pytorch versions support this, and we can drop internal support.

    """
    x = torch.randn(100, 10, dtype=torch.half)
    with pytest.raises(RuntimeError, match="\"topk_cpu\" not implemented for 'Half'"):
        torch.topk(x, k=3, dim=1)
