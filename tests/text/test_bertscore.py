import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.text.inputs import _inputs_multiple_sentences_multiple_reference
from torchmetrics.functional.text.bert import bert_score as metrics_bert_score
from torchmetrics.text.bert import BERTScore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE

if _BERTSCORE_AVAILABLE:
    from bert_score import score as original_bert_score

os.environ["TOKENIZERS_PARALLELISM"] = "1"

preds = _inputs_multiple_sentences_multiple_reference.preds
refs = _inputs_multiple_sentences_multiple_reference.targets

preds_batched = [preds[:2], preds[2:]]
refs_batched = [refs[:2], refs[2:]]


_METRICS = ["precision", "recall", "f1"]

MODEL_NAME = "albert-base-v2"


def _assert_list(preds: Any, refs: Any, threshold: float = 1e-8):
    """Assert two lists are equal."""
    assert np.allclose(preds, refs, atol=threshold, equal_nan=True)


def _parse_original_bert_score(score: torch.Tensor) -> Dict[str, List[float]]:
    """Parse the BERT score returned by the original `bert-score` package."""
    score_dict = {metric: value.tolist() for metric, value in zip(_METRICS, score)}
    return score_dict


@pytest.mark.parametrize(
    "refs,preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn(refs, preds):
    """Tests for functional."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs, preds, model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_with_idf(refs, preds):
    """Tests for functional with IDF rescaling."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, num_layers=12, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs, preds, model_name_or_path=MODEL_NAME, num_layers=12, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_all_layers(refs, preds):
    """Tests for functional and all layers."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, all_layers=True, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs, preds, model_name_or_path=MODEL_NAME, all_layers=True, idf=False, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_all_layers_with_idf(refs, preds):
    """Tests for functional and all layers with IDF rescaling."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs, preds, model_name_or_path=MODEL_NAME, all_layers=True, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_all_layers_rescale_with_baseline(preds, refs):
    """Tests for functional with baseline rescaling."""
    original_score = original_bert_score(
        preds,
        refs,
        model_type=MODEL_NAME,
        lang="en",
        num_layers=8,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs,
        preds,
        model_name_or_path=MODEL_NAME,
        lang="en",
        num_layers=8,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn_rescale_with_baseline(refs, preds):
    """Tests for functional with baseline rescaling with all layers."""
    original_score = original_bert_score(
        preds,
        refs,
        model_type=MODEL_NAME,
        lang="en",
        all_layers=True,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        refs,
        preds,
        model_name_or_path=MODEL_NAME,
        lang="en",
        all_layers=True,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score(refs, preds):
    """Tests for metric."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    Scorer.update(references=refs, predictions=preds)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_with_idf(refs, preds):
    """Tests for metric with IDF rescaling."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, num_layers=8, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=True, batch_size=3)
    Scorer.update(references=refs, predictions=preds)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_all_layers(refs, preds):
    """Tests for metric and all layers."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, all_layers=True, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path=MODEL_NAME, all_layers=True, idf=False, batch_size=3)
    Scorer.update(references=refs, predictions=preds)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_all_layers_with_idf(refs, preds):
    """Tests for metric and all layers with IDF rescaling."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    Scorer.update(references=refs, predictions=preds)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "refs, preds",
    [(refs_batched, preds_batched)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_accumulation(refs, preds):
    """Tests for metric works with accumulation."""
    original_score = original_bert_score(
        sum(preds, []), sum(refs, []), model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    Scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    for p, r in zip(preds, refs):
        Scorer.update(references=r, predictions=p)
    metrics_score = Scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


def _bert_score_ddp(rank, world_size, preds, refs, original_score):
    """Define a DDP process for BERTScore."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    Scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3, max_length=128)
    Scorer.update(refs, preds)
    metrics_score = Scorer.compute()
    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])
    dist.destroy_process_group()


def _test_score_ddp_fn(rank, world_size, preds, refs):
    """Core functionality for the `test_score_ddp` test."""
    original_score = original_bert_score(preds, refs, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)
    _bert_score_ddp(rank, world_size, preds, refs, original_score)


@pytest.mark.parametrize(
    "refs, preds",
    [(refs, preds)],
)
@pytest.mark.skipif(not (_BERTSCORE_AVAILABLE and dist.is_available()), reason="test requires bert_score")
def test_score_ddp(refs, preds):
    """Tests for metric using DDP."""
    world_size = 2
    mp.spawn(_test_score_ddp_fn, args=(world_size, preds, refs), nprocs=world_size, join=False)
