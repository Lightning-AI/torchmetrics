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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics.functional.text import squad
from torchmetrics.text.squad import SQuAD

from unittests.helpers.testers import _assert_allclose, _assert_tensor
from unittests.text.inputs import _inputs_squad_batch_match, _inputs_squad_exact_match, _inputs_squad_exact_mismatch


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_exact_match.preds,
            _inputs_squad_exact_match.target,
            _inputs_squad_exact_match.exact_match,
            _inputs_squad_exact_match.f1,
        ),
        (
            _inputs_squad_exact_mismatch.preds,
            _inputs_squad_exact_mismatch.target,
            _inputs_squad_exact_mismatch.exact_match,
            _inputs_squad_exact_mismatch.f1,
        ),
    ],
)
def test_score_fn(preds, targets, exact_match, f1):
    """Tests for functional."""
    metrics_score = squad(preds, targets)
    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], exact_match)
    _assert_allclose(metrics_score["f1"], f1)


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_batch_match.preds,
            _inputs_squad_batch_match.target,
            _inputs_squad_batch_match.exact_match,
            _inputs_squad_batch_match.f1,
        )
    ],
)
def test_accumulation(preds, targets, exact_match, f1):
    """Tests for metric works with accumulation."""
    squad_metric = SQuAD()
    for pred, target in zip(preds, targets):
        squad_metric.update(preds=[pred], target=[target])
    metrics_score = squad_metric.compute()

    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], torch.mean(torch.tensor(exact_match)))
    _assert_allclose(metrics_score["f1"], torch.mean(torch.tensor(f1)))


def _squad_score_ddp(rank, world_size, pred, targets, exact_match, f1):
    """Define a DDP process for SQuAD metric."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    squad_metric = SQuAD()
    squad_metric.update(pred, targets)
    metrics_score = squad_metric.compute()
    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], exact_match)
    _assert_allclose(metrics_score["f1"], f1)
    dist.destroy_process_group()


def _test_score_ddp_fn(rank, world_size, preds, targets, exact_match, f1):
    """Core functionality for the `test_score_ddp` test."""
    _squad_score_ddp(rank, world_size, preds[rank], targets[rank], exact_match[rank], f1[rank])


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_batch_match.preds,
            _inputs_squad_batch_match.target,
            _inputs_squad_batch_match.exact_match,
            _inputs_squad_batch_match.f1,
        )
    ],
)
@pytest.mark.skipif(not dist.is_available(), reason="test requires torch distributed")
def test_score_ddp(preds, targets, exact_match, f1):
    """Tests for metric using DDP."""
    world_size = 2
    mp.spawn(_test_score_ddp_fn, args=(world_size, preds, targets, exact_match, f1), nprocs=world_size, join=False)
