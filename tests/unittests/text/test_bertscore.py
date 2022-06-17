import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from torchmetrics.functional.text.bert import bert_score as metrics_bert_score
from torchmetrics.text.bert import BERTScore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE
from unittests.text.helpers import skip_on_connection_issues

if _BERTSCORE_AVAILABLE:
    from bert_score import score as original_bert_score

os.environ["TOKENIZERS_PARALLELISM"] = "1"

# Examples and expected values taken from:
# https://github.com/Tiiiger/bert_score/blob/master/tests/test_scorer.py
preds = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was "
    "found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at "
    'him."',
]
targets = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this "
    "week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally "
    'going well for him."',
]


_METRICS = ["precision", "recall", "f1"]

MODEL_NAME = "albert-base-v2"


def _assert_list(preds: Any, targets: Any, threshold: float = 1e-8):
    """Assert two lists are equal."""
    assert np.allclose(preds, targets, atol=threshold, equal_nan=True)


def _parse_original_bert_score(score: Tensor) -> Dict[str, List[float]]:
    """Parse the BERT score returned by the original `bert-score` package."""
    score_dict = {metric: value.tolist() for metric, value in zip(_METRICS, score)}
    return score_dict


preds_batched = [preds[0:2], preds[2:]]
targets_batched = [targets[0:2], targets[2:]]


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn(preds, targets):
    """Tests for functional."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, targets, model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn_with_idf(preds, targets):
    """Tests for functional with IDF rescaling."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, num_layers=12, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, targets, model_name_or_path=MODEL_NAME, num_layers=12, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.parametrize("device", ["cpu", "cuda", None])
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn_all_layers(preds, targets, device):
    """Tests for functional and all layers."""
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("Test requires GPU support")

    original_score = original_bert_score(
        preds, targets, model_type=MODEL_NAME, all_layers=True, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, targets, model_name_or_path=MODEL_NAME, all_layers=True, idf=False, batch_size=3, device=device
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn_all_layers_with_idf(preds, targets):
    """Tests for functional and all layers with IDF rescaling."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds, targets, model_name_or_path=MODEL_NAME, all_layers=True, idf=True, batch_size=3
    )

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn_all_layers_rescale_with_baseline(preds, targets):
    """Tests for functional with baseline rescaling."""
    original_score = original_bert_score(
        preds,
        targets,
        model_type=MODEL_NAME,
        lang="en",
        num_layers=8,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds,
        targets,
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
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_fn_rescale_with_baseline(preds, targets):
    """Tests for functional with baseline rescaling with all layers."""
    original_score = original_bert_score(
        preds,
        targets,
        model_type=MODEL_NAME,
        lang="en",
        all_layers=True,
        idf=False,
        batch_size=3,
        rescale_with_baseline=True,
    )
    original_score = _parse_original_bert_score(original_score)

    metrics_score = metrics_bert_score(
        preds,
        targets,
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
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score(preds, targets):
    """Tests for metric."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    scorer.update(preds=preds, target=targets)
    metrics_score = scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_with_idf(preds, targets):
    """Tests for metric with IDF rescaling."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, num_layers=8, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=True, batch_size=3)
    scorer.update(preds=preds, target=targets)
    metrics_score = scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_all_layers(preds, targets):
    """Tests for metric and all layers."""
    original_score = original_bert_score(
        preds, targets, model_type=MODEL_NAME, all_layers=True, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    scorer = BERTScore(model_name_or_path=MODEL_NAME, all_layers=True, idf=False, batch_size=3)
    scorer.update(preds=preds, target=targets)
    metrics_score = scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_all_layers_with_idf(preds, targets):
    """Tests for metric and all layers with IDF rescaling."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    original_score = _parse_original_bert_score(original_score)

    scorer = BERTScore(model_name_or_path=MODEL_NAME, all_layers=True, idf=True, batch_size=3)
    scorer.update(preds=preds, target=targets)
    metrics_score = scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


@pytest.mark.parametrize(
    "preds,targets",
    [(preds_batched, targets_batched)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def test_accumulation(preds, targets):
    """Tests for metric works with accumulation."""
    original_score = original_bert_score(
        sum(preds, []), sum(targets, []), model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3
    )
    original_score = _parse_original_bert_score(original_score)

    scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    for p, r in zip(preds, targets):
        scorer.update(preds=p, target=r)
    metrics_score = scorer.compute()

    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])


def _bert_score_ddp(rank, world_size, preds, targets, original_score):
    """Define a DDP process for BERTScore."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    scorer = BERTScore(model_name_or_path=MODEL_NAME, num_layers=8, idf=False, batch_size=3, max_length=128)
    scorer.update(preds, targets)
    metrics_score = scorer.compute()
    for metric in _METRICS:
        _assert_list(metrics_score[metric], original_score[metric])
    dist.destroy_process_group()


def _test_score_ddp_fn(rank, world_size, preds, targets):
    """Core functionality for the `test_score_ddp` test."""
    original_score = original_bert_score(preds, targets, model_type=MODEL_NAME, num_layers=8, idf=False, batch_size=3)
    original_score = _parse_original_bert_score(original_score)
    _bert_score_ddp(rank, world_size, preds, targets, original_score)


@pytest.mark.parametrize(
    "preds,targets",
    [(preds, targets)],
)
@pytest.mark.skipif(not (_BERTSCORE_AVAILABLE and dist.is_available()), reason="test requires bert_score")
@skip_on_connection_issues()
def test_score_ddp(preds, targets):
    """Tests for metric using DDP."""
    world_size = 2
    mp.spawn(_test_score_ddp_fn, args=(world_size, preds, targets), nprocs=world_size, join=False)
