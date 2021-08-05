from typing import Any

import numpy as np
import pytest

from torchmetrics.functional import bert_score
from torchmetrics.text import BERTScore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE

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


def _assert_list(preds: Any, refs: Any, threshold: float = 1e-8):
    """Assert two lists are equal."""
    assert np.allclose(preds, refs, atol=threshold, equal_nan=True)


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn(preds, refs):
    """Tests for functional."""
    Score = bert_score(preds, refs, model_type="roberta-large", num_layers=17, idf=False, batch_size=3)
    _assert_list(Score["precision"], [0.9843302369117737, 0.9832239747047424, 0.9120386242866516])
    _assert_list(Score["recall"], [0.9823839068412781, 0.9732863903045654, 0.920428991317749])
    _assert_list(Score["f1"], [0.9833561182022095, 0.9782299995422363, 0.916214644908905])


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score(preds, refs):
    """Tests for metric."""
    Scorer = BERTScore(model_type="roberta-large", num_layers=17, idf=False, batch_size=3)
    Scorer.update(predictions=preds, references=refs)
    Score = Scorer.compute()
    _assert_list(Score["precision"], [0.9843302369117737, 0.9832239747047424, 0.9120386242866516])
    _assert_list(Score["recall"], [0.9823839068412781, 0.9732863903045654, 0.920428991317749])
    _assert_list(Score["f1"], [0.9833561182022095, 0.9782299995422363, 0.916214644908905])
