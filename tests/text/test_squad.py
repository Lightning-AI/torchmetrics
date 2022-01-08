import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.testers import _assert_allclose, _assert_tensor
from torchmetrics.functional.text import squad
from torchmetrics.text.squad import SQuAD

SAMPLE_1 = {
    "exact_match": 100.0,
    "f1": 100.0,
    "predictions": {"prediction_text": "1976", "id": "id1"},
    "references": {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
}

SAMPLE_2 = {
    "exact_match": 0.0,
    "f1": 0.0,
    "predictions": {"prediction_text": "Hello", "id": "id2"},
    "references": {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
}

BATCH = {
    "exact_match": [100.0, 0.0],
    "f1": [100.0, 0.0],
    "predictions": [
        {"prediction_text": "1976", "id": "id1"},
        {"prediction_text": "Hello", "id": "id2"},
    ],
    "references": [
        {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
        {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
    ],
}


@pytest.mark.parametrize(
    "preds,targets,exact_match,f1",
    [
        (SAMPLE_1["predictions"], SAMPLE_1["references"], SAMPLE_1["exact_match"], SAMPLE_1["exact_match"]),
        (SAMPLE_2["predictions"], SAMPLE_2["references"], SAMPLE_2["exact_match"], SAMPLE_2["exact_match"]),
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
    "preds,targets,exact_match,f1",
    [(BATCH["predictions"], BATCH["references"], BATCH["exact_match"], BATCH["f1"])],
)
def test_accumulation(preds, targets, exact_match, f1):
    """Tests for metric works with accumulation."""
    squad_metric = SQuAD()
    for pred, target in zip(preds, targets):
        squad_metric.update(preds=[pred], targets=[target])
    metrics_score = squad_metric.compute()

    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], torch.mean(torch.tensor(exact_match)))
    _assert_allclose(metrics_score["f1"], torch.mean(torch.tensor(f1)))


def _squad_score_ddp(rank, world_size, pred, target, exact_match, f1):
    """Define a DDP process for SQuAD metric."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    squad_metric = SQuAD()
    squad_metric.update(pred, target)
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
    "preds,targets,exact_match,f1",
    [(BATCH["predictions"], BATCH["references"], BATCH["exact_match"], BATCH["f1"])],
)
@pytest.mark.skipif(not dist.is_available(), reason="test requires torch distributed")
def test_score_ddp(preds, targets, exact_match, f1):
    """Tests for metric using DDP."""
    world_size = 2
    mp.spawn(_test_score_ddp_fn, args=(world_size, preds, targets, exact_match, f1), nprocs=world_size, join=False)
