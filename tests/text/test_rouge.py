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

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import BootstrapAggregator
else:
    RougeScorer, BootstrapAggregator = object, object

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

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
        aggregator.add_scores(scorer.score(target, pred))
    return aggregator.aggregate()


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer"],
    [
        pytest.param("rouge1_precision", True),
        pytest.param("rouge1_recall", True),
        pytest.param("rouge1_fmeasure", False),
        pytest.param("rouge2_precision", False),
        pytest.param("rouge2_recall", True),
        pytest.param("rouge2_fmeasure", True),
        pytest.param("rougeL_precision", False),
        pytest.param("rougeL_recall", False),
        pytest.param("rougeL_fmeasure", True),
        pytest.param("rougeLsum_precision", True),
        pytest.param("rougeLsum_recall", False),
        pytest.param("rougeLsum_fmeasure", False),
    ],
)
def test_rouge_metric_functional_single_sentence(pl_rouge_metric_key, use_stemmer):
    rouge_level, metric = pl_rouge_metric_key.split("_")

    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    rs_scores = scorer.score(SINGLE_SENTENCE_EXAMPLE_TARGET, SINGLE_SENTENCE_EXAMPLE_PREDS)
    rs_result = torch.tensor(getattr(rs_scores[rouge_level], metric), dtype=torch.float32)

    pl_output = rouge_score([SINGLE_SENTENCE_EXAMPLE_PREDS], [SINGLE_SENTENCE_EXAMPLE_TARGET], use_stemmer=use_stemmer)

    assert torch.allclose(pl_output[pl_rouge_metric_key], rs_result)


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer"],
    [
        pytest.param("rouge1_precision", True),
        pytest.param("rouge1_recall", True),
        pytest.param("rouge1_fmeasure", False),
        pytest.param("rouge2_precision", False),
        pytest.param("rouge2_recall", True),
        pytest.param("rouge2_fmeasure", True),
        pytest.param("rougeL_precision", False),
        pytest.param("rougeL_recall", False),
        pytest.param("rougeL_fmeasure", True),
        pytest.param("rougeLsum_precision", True),
        pytest.param("rougeLsum_recall", False),
        pytest.param("rougeLsum_fmeasure", False),
    ],
)
def test_rouge_metric_functional(pl_rouge_metric_key, use_stemmer):
    rouge_level, metric = pl_rouge_metric_key.split("_")

    rs_scores = _compute_rouge_score(PREDS, TARGETS, use_stemmer=use_stemmer)
    rs_result = torch.tensor(
        getattr(rs_scores[rouge_level].mid, metric),
        dtype=torch.float32,
    )

    pl_output = rouge_score(PREDS, TARGETS, use_stemmer=use_stemmer)

    assert torch.allclose(pl_output[pl_rouge_metric_key], rs_result)


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer"],
    [
        pytest.param("rouge1_precision", True),
        pytest.param("rouge1_recall", True),
        pytest.param("rouge1_fmeasure", False),
        pytest.param("rouge2_precision", False),
        pytest.param("rouge2_recall", True),
        pytest.param("rouge2_fmeasure", True),
        pytest.param("rougeL_precision", False),
        pytest.param("rougeL_recall", False),
        pytest.param("rougeL_fmeasure", True),
        pytest.param("rougeLsum_precision", True),
        pytest.param("rougeLsum_recall", False),
        pytest.param("rougeLsum_fmeasure", False),
    ],
)
def test_rouge_metric_class(pl_rouge_metric_key, use_stemmer):
    rouge_level, metric = pl_rouge_metric_key.split("_")

    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    rs_scores = scorer.score(SINGLE_SENTENCE_EXAMPLE_TARGET, SINGLE_SENTENCE_EXAMPLE_PREDS)
    rs_result = torch.tensor(getattr(rs_scores[rouge_level], metric), dtype=torch.float32)

    rouge = ROUGEScore(use_stemmer=use_stemmer)
    pl_output = rouge([SINGLE_SENTENCE_EXAMPLE_PREDS], [SINGLE_SENTENCE_EXAMPLE_TARGET])

    assert torch.allclose(pl_output[pl_rouge_metric_key], rs_result)


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer"],
    [
        pytest.param("rouge1_precision", True),
        pytest.param("rouge1_recall", True),
        pytest.param("rouge1_fmeasure", False),
        pytest.param("rouge2_precision", False),
        pytest.param("rouge2_recall", True),
        pytest.param("rouge2_fmeasure", True),
        pytest.param("rougeL_precision", False),
        pytest.param("rougeL_recall", False),
        pytest.param("rougeL_fmeasure", True),
        pytest.param("rougeLsum_precision", True),
        pytest.param("rougeLsum_recall", False),
        pytest.param("rougeLsum_fmeasure", False),
    ],
)
def test_rouge_metric_class_batches(pl_rouge_metric_key, use_stemmer):
    rouge_level, metric = pl_rouge_metric_key.split("_")

    rs_scores = _compute_rouge_score(BATCHES_RS_PREDS, BATCHES_RS_TARGETS, use_stemmer=use_stemmer)
    rs_result = torch.tensor(getattr(rs_scores[rouge_level].mid, metric), dtype=torch.float32)

    rouge = ROUGEScore(use_stemmer=use_stemmer)
    for batch in BATCHES:
        rouge.update(batch["preds"], batch["targets"])
    pl_output = rouge.compute()

    assert torch.allclose(pl_output[pl_rouge_metric_key], rs_result)


def test_rouge_metric_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    if not _NLTK_AVAILABLE:
        with pytest.raises(
            ValueError,
            match="ROUGE metric requires that nltk is installed."
            "Either as `pip install torchmetrics[text]` or `pip install nltk`",
        ):
            ROUGEScore()


def test_rouge_metric_wrong_key_value_error():
    key = ("rouge1", "rouge")

    with pytest.raises(ValueError):
        ROUGEScore(rouge_keys=key)

    with pytest.raises(ValueError):
        rouge_score(PREDS, TARGETS, rouge_keys=key)
