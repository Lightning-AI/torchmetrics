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
import os
import sys
from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import tensor

from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import seed_all
from unittests._helpers.testers import DummyListMetric, DummyMetric, DummyMetricSum
from unittests.conftest import setup_ddp

seed_all(42)


def _test_ddp_sum(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = tensor(1)
    dummy._sync_dist()

    assert dummy.foo == worldsize


def _test_ddp_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [tensor([1])]
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))


def _test_ddp_sum_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat, "bar": torch.sum}
    dummy.foo = [tensor([1])]
    dummy.bar = tensor(1)
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))
    assert dummy.bar == worldsize


def _test_ddp_gather_uneven_tensors(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    tensor = torch.ones(rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_ddp_gather_uneven_tensors_multidim(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    tensor = torch.ones(rank + 1, 2 - rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert (val == torch.ones_like(val)).all()


def _test_ddp_compositional_tensor(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetricSum()
    dummy._reductions = {"x": torch.sum}
    dummy = dummy.clone() + dummy.clone()
    dummy.update(tensor(1))
    val = dummy.compute()
    assert val == 2 * worldsize


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(
    "process",
    [
        _test_ddp_cat,
        _test_ddp_sum,
        _test_ddp_sum_cat,
        _test_ddp_gather_uneven_tensors,
        _test_ddp_gather_uneven_tensors_multidim,
        _test_ddp_compositional_tensor,
    ],
)
def test_ddp(process):
    """Test ddp functions."""
    pytest.pool.map(process, range(NUM_PROCESSES))


def _test_ddp_gather_all_autograd_same_shape(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    """Test that ddp gather preserves local rank's autograd graph for same-shaped tensors across ranks."""
    setup_ddp(rank, worldsize)
    x = (rank + 1) * torch.ones(10, requires_grad=True)

    # random linear transformation, it should really not matter what we do here
    a, b = torch.randn(1), torch.randn(1)
    y = a * x + b  # gradient of y w.r.t. x is a

    result = gather_all_tensors(y)
    assert len(result) == worldsize
    grad = torch.autograd.grad(result[rank].sum(), x)[0]
    assert torch.allclose(grad, a * torch.ones_like(x))


def _test_ddp_gather_all_autograd_different_shape(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    """Test that ddp gather preserves local rank's autograd graph for differently-shaped tensors across ranks."""
    setup_ddp(rank, worldsize)
    x = (rank + 1) * torch.ones(rank + 1, 2 - rank, requires_grad=True)

    # random linear transformation, it should really not matter what we do here
    a, b = torch.randn(1), torch.randn(1)
    y = a * x + b  # gradient of y w.r.t. x is a

    result = gather_all_tensors(y)
    assert len(result) == worldsize
    grad = torch.autograd.grad(result[rank].sum(), x)[0]
    assert torch.allclose(grad, a * torch.ones_like(x))


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(
    "process", [_test_ddp_gather_all_autograd_same_shape, _test_ddp_gather_all_autograd_different_shape]
)
def test_ddp_autograd(process):
    """Test ddp functions for autograd compatibility."""
    pytest.pool.map(process, range(NUM_PROCESSES))


def _test_non_contiguous_tensors(rank):
    class DummyCatMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()
            self.add_state("x", default=[], dist_reduce_fx=None)

        def update(self, x):
            self.x.append(x)

        def compute(self):
            x = torch.cat(self.x, dim=0)
            return x.sum()

    metric = DummyCatMetric()
    metric.update(torch.randn(10, 5)[:, 0])


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_non_contiguous_tensors():
    """Test that gather_all operation works for non-contiguous tensors."""
    pytest.pool.map(_test_non_contiguous_tensors, range(NUM_PROCESSES))


def _test_state_dict_is_synced(rank, tmpdir):
    class DummyCatMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()
            self.add_state("x", torch.tensor(0), dist_reduce_fx=torch.sum)
            self.add_state("c", torch.tensor(0), dist_reduce_fx=torch.sum)

        def update(self, x):
            self.x += x
            self.c += 1

        def compute(self):
            return self.x // self.c

        def __repr__(self) -> str:
            return f"DummyCatMetric(x={self.x}, c={self.c})"

    metric = DummyCatMetric()
    metric.persistent(True)

    def verify_metric(metric, i, world_size):
        state_dict = metric.state_dict()
        exp_sum = i * (i + 1) / 2
        assert state_dict["x"] == exp_sum * world_size
        assert metric.x == exp_sum * world_size
        assert metric.c == (i + 1) * world_size
        assert state_dict["c"] == metric.c

    steps = 5
    for i in range(steps):
        if metric._is_synced:
            with pytest.raises(TorchMetricsUserError, match="The Metric shouldn't be synced when performing"):
                metric(i)

            metric.unsync()

        metric(i)

        verify_metric(metric, i, 1)

        metric.sync()
        assert metric._is_synced

        with pytest.raises(TorchMetricsUserError, match="The Metric has already been synced."):
            metric.sync()

        verify_metric(metric, i, 2)

        metric.unsync()
        assert not metric._is_synced

        with pytest.raises(TorchMetricsUserError, match="The Metric has already been un-synced."):
            metric.unsync()

        with metric.sync_context():
            assert metric._is_synced
            verify_metric(metric, i, 2)

        with metric.sync_context(should_unsync=False):
            assert metric._is_synced
            verify_metric(metric, i, 2)

        assert metric._is_synced

        metric.unsync()
        assert not metric._is_synced

        metric.sync()
        cache = metric._cache
        metric._cache = None

        with pytest.raises(TorchMetricsUserError, match="The internal cache should exist to unsync the Metric."):
            metric.unsync()

        metric._cache = cache

    def reload_state_dict(state_dict, expected_x, expected_c):
        metric = DummyCatMetric()
        metric.load_state_dict(state_dict)
        assert metric.x == expected_x
        assert metric.c == expected_c

    reload_state_dict(deepcopy(metric.state_dict()), 20, 10)

    metric.unsync()
    reload_state_dict(deepcopy(metric.state_dict()), 10, 5)

    metric.sync()

    filepath = os.path.join(tmpdir, f"weights-{rank}.pt")

    torch.save(metric.state_dict(), filepath)

    metric.unsync()
    with metric.sync_context():
        torch.save(metric.state_dict(), filepath)


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_state_dict_is_synced(tmpdir):
    """Tests that metrics are synced while creating the state dict but restored after to continue accumulation."""
    pytest.pool.map(partial(_test_state_dict_is_synced, tmpdir=tmpdir), range(NUM_PROCESSES))


def _test_sync_on_compute_tensor_state(rank, sync_on_compute):
    dummy = DummyMetricSum(sync_on_compute=sync_on_compute)
    dummy.update(tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert val == 3
    else:
        assert val == rank + 1


def _test_sync_on_compute_list_state(rank, sync_on_compute):
    dummy = DummyListMetric(sync_on_compute=sync_on_compute)
    dummy.update(tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert val.sum() == 3
        assert torch.allclose(val, tensor([1, 2])) or torch.allclose(val, tensor([2, 1]))
    else:
        assert val == [tensor(rank + 1)]


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize("sync_on_compute", [True, False])
@pytest.mark.parametrize("test_func", [_test_sync_on_compute_list_state, _test_sync_on_compute_tensor_state])
def test_sync_on_compute(sync_on_compute, test_func):
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(partial(test_func, sync_on_compute=sync_on_compute), range(NUM_PROCESSES))


def _test_sync_with_empty_lists(rank):
    dummy = DummyListMetric()
    val = dummy.compute()
    assert torch.allclose(val, tensor([]))


@pytest.mark.DDP
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_2_1, reason="test only works on newer torch versions")
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_sync_with_empty_lists():
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(_test_sync_with_empty_lists, range(NUM_PROCESSES))


def _test_sync_with_unequal_size_lists(rank):
    """Test that synchronization of list states work even when some ranks have not received any data yet."""
    dummy = DummyListMetric()
    if rank == 0:
        dummy.update(torch.zeros(2))
    assert torch.all(dummy.compute() == tensor([0.0, 0.0]))


@pytest.mark.DDP
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_2_1, reason="test only works on newer torch versions")
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_sync_with_unequal_size_lists():
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(_test_sync_with_unequal_size_lists, range(NUM_PROCESSES))
