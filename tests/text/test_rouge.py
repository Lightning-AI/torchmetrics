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

from typing import List

import pytest
import torch
from torch import tensor

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import BootstrapAggregator
else:
    RougeScorer, BootstrapAggregator = object, object

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

PRECISION = 0
RECALL = 1
F_MEASURE = 2

SINGLE_SENTENCE_EXAMPLE_PREDS = "The quick brown fox jumps over the lazy dog"
SINGLE_SENTENCE_EXAMPLE_TARGET = "The quick brown dog jumps on the log."

PREDS = "My name is John".split()
TARGETS = "Is your name John".split()

BATCHES_RS_PREDS = [SINGLE_SENTENCE_EXAMPLE_PREDS]
BATCHES_RS_PREDS.extend(PREDS)
BATCHES_RS_TARGETS = [SINGLE_SENTENCE_EXAMPLE_TARGET]
BATCHES_RS_TARGETS.extend(TARGETS)

BATCHES = [
    dict(preds=[SINGLE_SENTENCE_EXAMPLE_PREDS], targets=[SINGLE_SENTENCE_EXAMPLE_TARGET]),
    dict(preds=PREDS, targets=TARGETS),
]


def _compute_rouge_score(preds: List[str], targets: List[str], use_stemmer: bool):
    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = BootstrapAggregator()
    for pred, target in zip(preds, targets):
        aggregator.add_scores(scorer.score(pred, target))
    return aggregator.aggregate()


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason="test requires nltk and rouge-score")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "rouge_score_key", "metric", "decimal_places", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", "rouge1", PRECISION, 1, True, True),
        pytest.param("rouge1_recall", "rouge1", RECALL, 2, True, False),
        pytest.param("rouge1_fmeasure", "rouge1", F_MEASURE, 3, False, True),
        pytest.param("rouge2_precision", "rouge2", PRECISION, 4, False, False),
        pytest.param("rouge2_recall", "rouge2", RECALL, 5, True, True),
        pytest.param("rouge2_fmeasure", "rouge2", F_MEASURE, 6, True, False),
        pytest.param("rougeL_precision", "rougeL", PRECISION, 6, False, True),
        pytest.param("rougeL_recall", "rougeL", RECALL, 5, False, False),
        pytest.param("rougeL_fmeasure", "rougeL", F_MEASURE, 3, True, True),
        pytest.param("rougeLsum_precision", "rougeLsum", PRECISION, 2, True, False),
        pytest.param("rougeLsum_recall", "rougeLsum", RECALL, 1, False, True),
        pytest.param("rougeLsum_fmeasure", "rougeLsum", F_MEASURE, 8, False, False),
    ],
)
def test_rouge_metric_functional_single_sentence(
    pl_rouge_metric_key, rouge_score_key, metric, decimal_places, use_stemmer, newline_sep
):
    scorer = RougeScorer(ROUGE_KEYS)
    rs_scores = scorer.score(SINGLE_SENTENCE_EXAMPLE_PREDS, SINGLE_SENTENCE_EXAMPLE_TARGET)
    rs_output = round(rs_scores[rouge_score_key][metric], decimal_places)

    pl_output = rouge_score(
        [SINGLE_SENTENCE_EXAMPLE_PREDS],
        [SINGLE_SENTENCE_EXAMPLE_TARGET],
        newline_sep=newline_sep,
        use_stemmer=use_stemmer,
        decimal_places=decimal_places,
    )

    assert torch.allclose(pl_output[pl_rouge_metric_key], tensor(rs_output, dtype=torch.float32))


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason="test requires nltk and rouge-score")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "rouge_score_key", "metric", "decimal_places", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", "rouge1", PRECISION, 1, True, True),
        pytest.param("rouge1_recall", "rouge1", RECALL, 2, True, False),
        pytest.param("rouge1_fmeasure", "rouge1", F_MEASURE, 3, False, True),
        pytest.param("rouge2_precision", "rouge2", PRECISION, 4, False, False),
        pytest.param("rouge2_recall", "rouge2", RECALL, 5, True, True),
        pytest.param("rouge2_fmeasure", "rouge2", F_MEASURE, 6, True, False),
        pytest.param("rougeL_precision", "rougeL", PRECISION, 6, False, True),
        pytest.param("rougeL_recall", "rougeL", RECALL, 5, False, False),
        pytest.param("rougeL_fmeasure", "rougeL", F_MEASURE, 3, True, True),
        pytest.param("rougeLsum_precision", "rougeLsum", PRECISION, 2, True, False),
        pytest.param("rougeLsum_recall", "rougeLsum", RECALL, 1, False, True),
        pytest.param("rougeLsum_fmeasure", "rougeLsum", F_MEASURE, 8, False, False),
    ],
)
def test_rouge_metric_functional(
    pl_rouge_metric_key, rouge_score_key, metric, decimal_places, use_stemmer, newline_sep
):
    rs_scores = _compute_rouge_score(PREDS, TARGETS, use_stemmer=use_stemmer)
    rs_output = round(rs_scores[rouge_score_key].mid[metric], decimal_places)

    pl_output = rouge_score(
        PREDS, TARGETS, newline_sep=newline_sep, use_stemmer=use_stemmer, decimal_places=decimal_places
    )

    assert torch.allclose(pl_output[pl_rouge_metric_key], tensor(rs_output, dtype=torch.float32))


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason="test requires nltk and rouge-score")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "rouge_score_key", "metric", "decimal_places", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", "rouge1", PRECISION, 1, True, True),
        pytest.param("rouge1_recall", "rouge1", RECALL, 2, True, False),
        pytest.param("rouge1_fmeasure", "rouge1", F_MEASURE, 3, False, True),
        pytest.param("rouge2_precision", "rouge2", PRECISION, 4, False, False),
        pytest.param("rouge2_recall", "rouge2", RECALL, 5, True, True),
        pytest.param("rouge2_fmeasure", "rouge2", F_MEASURE, 6, True, False),
        pytest.param("rougeL_precision", "rougeL", PRECISION, 6, False, True),
        pytest.param("rougeL_recall", "rougeL", RECALL, 5, False, False),
        pytest.param("rougeL_fmeasure", "rougeL", F_MEASURE, 3, True, True),
        pytest.param("rougeLsum_precision", "rougeLsum", PRECISION, 2, True, False),
        pytest.param("rougeLsum_recall", "rougeLsum", RECALL, 1, False, True),
        pytest.param("rougeLsum_fmeasure", "rougeLsum", F_MEASURE, 8, False, False),
    ],
)
def test_rouge_metric_class(pl_rouge_metric_key, rouge_score_key, metric, decimal_places, use_stemmer, newline_sep):
    scorer = RougeScorer(ROUGE_KEYS)
    rs_scores = scorer.score(SINGLE_SENTENCE_EXAMPLE_PREDS, SINGLE_SENTENCE_EXAMPLE_TARGET)
    rs_output = round(rs_scores[rouge_score_key][metric], decimal_places)

    rouge = ROUGEScore(newline_sep=newline_sep, use_stemmer=use_stemmer, decimal_places=decimal_places)
    pl_output = rouge([SINGLE_SENTENCE_EXAMPLE_PREDS], [SINGLE_SENTENCE_EXAMPLE_TARGET])

    assert torch.allclose(pl_output[pl_rouge_metric_key], tensor(rs_output, dtype=torch.float32))


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason="test requires nltk and rouge-score")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "rouge_score_key", "metric", "decimal_places", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", "rouge1", PRECISION, 1, True, True),
        pytest.param("rouge1_recall", "rouge1", RECALL, 2, True, False),
        pytest.param("rouge1_fmeasure", "rouge1", F_MEASURE, 3, False, True),
        pytest.param("rouge2_precision", "rouge2", PRECISION, 4, False, False),
        pytest.param("rouge2_recall", "rouge2", RECALL, 5, True, True),
        pytest.param("rouge2_fmeasure", "rouge2", F_MEASURE, 6, True, False),
        pytest.param("rougeL_precision", "rougeL", PRECISION, 6, False, True),
        pytest.param("rougeL_recall", "rougeL", RECALL, 5, False, False),
        pytest.param("rougeL_fmeasure", "rougeL", F_MEASURE, 3, True, True),
        pytest.param("rougeLsum_precision", "rougeLsum", PRECISION, 2, True, False),
        pytest.param("rougeLsum_recall", "rougeLsum", RECALL, 1, False, True),
        pytest.param("rougeLsum_fmeasure", "rougeLsum", F_MEASURE, 8, False, False),
    ],
)
def test_rouge_metric_class_batches(
    pl_rouge_metric_key, rouge_score_key, metric, decimal_places, use_stemmer, newline_sep
):
    rs_scores = _compute_rouge_score(BATCHES_RS_PREDS, BATCHES_RS_TARGETS, use_stemmer=use_stemmer)
    rs_output = round(rs_scores[rouge_score_key].mid[metric], decimal_places)

    rouge = ROUGEScore(newline_sep=newline_sep, use_stemmer=use_stemmer, decimal_places=decimal_places)
    for batch in BATCHES:
        rouge.update(batch["preds"], batch["targets"])
    pl_output = rouge.compute()

    assert torch.allclose(pl_output[pl_rouge_metric_key], tensor(rs_output, dtype=torch.float32))


def test_rouge_metric_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised"""
    if not (_NLTK_AVAILABLE and _ROUGE_SCORE_AVAILABLE):
        with pytest.raises(
            ValueError,
            match="ROUGE metric requires that both nltk and rouge-score is installed."
            "Either as `pip install torchmetrics[text]` or `pip install nltk rouge-score`",
        ):
            ROUGEScore()


def test_rouge_metric_wrong_key_value_error():
    key = ("rouge1", "rouge")

    with pytest.raises(ValueError):
        ROUGEScore(rouge_keys=key)

    with pytest.raises(ValueError):
        rouge_score(PREDS, TARGETS, rouge_keys=key)
