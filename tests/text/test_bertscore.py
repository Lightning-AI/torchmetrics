import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchmetrics.functional import bert_score as metrics_bert_score
from torchmetrics.text import BERTScore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE

if _BERTSCORE_AVAILABLE:
    from bert_score import score as original_bert_score

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Examples and expected values taken from:
# https://github.com/Tiiiger/bert_score/blob/master/tests/test_scorer.py
preds = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was "
    "found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at "
    'him."',
]
refs = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this "
    "week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally "
    'going well for him."',
]


_METRICS = ["precision", "recall", "f1"]


def _assert_list(preds: Any, refs: Any, threshold: float = 1e-8):
    """Assert two lists are equal."""
    assert np.allclose(preds, refs, atol=threshold, equal_nan=True)


def _parse_original_bert_score(score: torch.Tensor) -> Dict[str, List[float]]:
    score_dict = {metric: value.tolist() for metric, value in zip(_METRICS, score)}
    return score_dict


preds_batched = [preds[0:2], preds[2:]]
refs_batched = [refs[0:2], refs[2:]]


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn(preds, refs):
    """Tests for functional."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, refs, model_name_or_path="bert-base-uncased", num_layers=8, idf=False, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_with_idf(preds, refs):
    """Tests for functional with IDF rescaling."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", num_layers=12, idf=True, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, refs, model_name_or_path="bert-base-uncased", num_layers=12, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_all_layers(preds, refs):
    """Tests for functional and all layers."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", all_layers=True, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, refs, model_name_or_path="bert-base-uncased", all_layers=True, idf=False, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_all_layers_with_idf(preds, refs):
    """Tests for functional and all layers with IDF rescaling."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", all_layers=True, idf=True, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, refs, model_name_or_path="bert-base-uncased", all_layers=True, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score(preds, refs):
    """Tests for metric."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path="bert-base-uncased", num_layers=8, idf=False, batch_size=3)
    Scorer.update(predictions=preds, references=refs)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_with_idf(preds, refs):
    """Tests for metric with IDF rescaling."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", num_layers=8, idf=True, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path="bert-base-uncased", num_layers=8, idf=True, batch_size=3)
    Scorer.update(predictions=preds, references=refs)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_all_layers(preds, refs):
    """Tests for metric and all layers."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", all_layers=True, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path="bert-base-uncased", all_layers=True, idf=False, batch_size=3)
    Scorer.update(predictions=preds, references=refs)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_all_layers_with_idf(preds, refs):
    """Tests for metric and all layers with IDF rescaling."""
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", all_layers=True, idf=True, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path="bert-base-uncased", all_layers=True, idf=True, batch_size=3)
    Scorer.update(predictions=preds, references=refs)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds_batched, refs_batched)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_accumulation(preds, refs):
    """Tests for metric works with accumulation."""
    original_score = original_bert_score(
        sum(preds, []), sum(refs, []), model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path="bert-base-uncased", num_layers=8, idf=False, batch_size=3)
    for p, r in zip(preds, refs):
        Scorer.update(predictions=p, references=r)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


def _bert_score_ddp(rank, world_size, preds, refs, original_score):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    Scorer = BERTScore(model_name_or_path="bert-base-uncased", num_layers=8, idf=False, batch_size=3)
    Scorer.update(preds, refs)
    metrics_score = Scorer.compute()
    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])
    dist.destroy_process_group()


def _test_score_ddp_fn(rank, world_size, preds, refs):
    original_score = original_bert_score(
        preds, refs, model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)
    _bert_score_ddp(rank, world_size, preds, refs, original_score)


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_ddp(preds, refs):
    """Tests for metric using DDP."""
    world_size = 2
    mp.spawn(_test_score_ddp_fn, args=(world_size, preds, refs), nprocs=world_size, join=True)
