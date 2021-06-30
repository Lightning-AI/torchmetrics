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
    "process", [
        _test_ddp_cat,
        _test_ddp_sum,
        _test_ddp_sum_cat,
        _test_ddp_gather_uneven_tensors,
        _test_ddp_gather_uneven_tensors_multidim,
    ]
)
def test_ddp(process):
    torch.multiprocessing.spawn(process, args=(2, ), nprocs=2)


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
    """ Test that gather_all operation works for non contiguous tensors """
    torch.multiprocessing.spawn(_test_non_contiguous_tensors, args=(2, ), nprocs=2)


def _test_state_dict_is_synced(rank, worldsize, tmpdir):
    setup_ddp(rank, worldsize)

    is_global_zero = not rank

    class DummyCatMetric(Metric):

        def __init__(self, should_sync_state_dict: bool = True):
            super().__init__(should_sync_state_dict=should_sync_state_dict)
            self.add_state("x", torch.tensor(0), dist_reduce_fx=torch.sum)
            self.add_state("c", torch.tensor(0), dist_reduce_fx=torch.sum)

        def update(self, x):
            self.x += x
            self.c += 1

        def compute(self):
            return self.x // self.c

    metric = DummyCatMetric()
    metric.persistent(True)

    metric_2 = DummyCatMetric(should_sync_state_dict=False)
    metric.persistent(True)

    steps = 5
    for i in range(steps):
        metric(i)
        metric_2(i)
        state_dict = metric.state_dict()
        state_dict_not_synced = metric.state_dict(should_sync=False)

        state_dict_2_should_not_sync = metric_2.state_dict()
        state_dict_2_should_sync = metric_2.state_dict(should_sync=True)

        exp_sum = i * (i + 1) / 2
        assert metric.x == exp_sum
        assert metric.c == (i + 1)
        if rank == 0:
            assert state_dict["x"] == exp_sum * worldsize
            assert state_dict["c"] == metric.c * worldsize
        else:
            assert state_dict["x"] == 0
            assert state_dict["c"] == 0 


    assert state_dict["has_synced"]
    assert state_dict_2_should_sync["has_synced"]
    assert not state_dict_2_should_not_sync["has_synced"]
    assert not state_dict_not_synced["has_synced"]

    def reload_state_dict(state_dict, expected_x, expected_c):
        metric = DummyCatMetric()
        metric.load_state_dict(state_dict)
        assert metric.x == expected_x
        assert metric.c == expected_c

    reload_state_dict(deepcopy(state_dict), 10 + 10 if is_global_zero else 0, 5 + 5 if is_global_zero else 0)
    reload_state_dict(deepcopy(state_dict_not_synced), 10, 5)

    path = os.path.join(tmpdir, "metric.p")
    if rank == 0:
        torch.save(state_dict_not_synced, path)

    path_1 = os.path.join(tmpdir, "metric_1.p")
    if rank == 1:
        torch.save(state_dict_not_synced, path_1)

    torch.distributed.barrier()

    state_dict_not_synced = torch.load(path)

    reload_state_dict(deepcopy(state_dict_not_synced), 10 if is_global_zero else 0, 5 if is_global_zero else 0)

    reload_state_dict(deepcopy(state_dict_not_synced), 10 if is_global_zero else 0, 5 if is_global_zero else 0)

    # shows that multiple checkpoints can be reloaded properly.
    state_dict_not_synced = torch.load(path_1)
    reload_state_dict(deepcopy(state_dict_not_synced), 10, 5)



@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_state_dict_is_synced(tmpdir):
    """
    This test asserts that metrics are synced while creating the state
    dict but restored after to continue accumulation.
    """
    torch.multiprocessing.spawn(_test_state_dict_is_synced, args=(2, tmpdir), nprocs=2)
