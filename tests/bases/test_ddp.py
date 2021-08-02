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
import os
import sys
from copy import deepcopy

import pytest
import torch
from torch import tensor

from tests.helpers import seed_all
from tests.helpers.testers import DummyMetric, setup_ddp
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError

seed_all(42)


def _test_ddp_sum(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = tensor(1)
    dummy._sync_dist()

    assert dummy.foo == worldsize


def _test_ddp_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [tensor([1])]
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))


def _test_ddp_sum_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat, "bar": torch.sum}
    dummy.foo = [tensor([1])]
    dummy.bar = tensor(1)
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))
    assert dummy.bar == worldsize


def _test_ddp_gather_uneven_tensors(rank, worldsize):
    setup_ddp(rank, worldsize)
    tensor = torch.ones(rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        assert len(result[idx]) == idx
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_ddp_gather_uneven_tensors_multidim(rank, worldsize):
    setup_ddp(rank, worldsize)
    tensor = torch.ones(rank + 1, 2 - rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert val.shape == (idx + 1, 2 - idx)
        assert (val == torch.ones_like(val)).all()


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize(
    "process",
    [
        _test_ddp_cat,
        _test_ddp_sum,
        _test_ddp_sum_cat,
        _test_ddp_gather_uneven_tensors,
        _test_ddp_gather_uneven_tensors_multidim,
    ],
)
def test_ddp(process):
    torch.multiprocessing.spawn(process, args=(2,), nprocs=2)


def _test_non_contiguous_tensors(rank, worldsize):
    setup_ddp(rank, worldsize)

    class DummyCatMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("x", default=[], dist_reduce_fx=None)

        def update(self, x):
            self.x.append(x)

        def compute(self):
            x = torch.cat(self.x, dim=0)
            return x.sum()

    metric = DummyCatMetric()
    metric.update(torch.randn(10, 5)[:, 0])


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_non_contiguous_tensors():
    """Test that gather_all operation works for non contiguous tensors"""
    torch.multiprocessing.spawn(_test_non_contiguous_tensors, args=(2,), nprocs=2)


def _test_state_dict_is_synced(rank, worldsize, tmpdir):
    setup_ddp(rank, worldsize)

    class DummyCatMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("x", torch.tensor(0), dist_reduce_fx=torch.sum)
            self.add_state("c", torch.tensor(0), dist_reduce_fx=torch.sum)

        def update(self, x):
            self.x += x
            self.c += 1

        def compute(self):
            return self.x // self.c

        def __repr__(self):
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


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_state_dict_is_synced(tmpdir):
    """
    This test asserts that metrics are synced while creating the state
    dict but restored after to continue accumulation.
    """
    torch.multiprocessing.spawn(_test_state_dict_is_synced, args=(2, tmpdir), nprocs=2)
