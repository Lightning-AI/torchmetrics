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

from functools import partial
from typing import List

import pytest

from tests.text.helpers import INPUT_ORDER, TextTester
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

PREDS = "My name is John"
TARGETS = "Is your name John"


BATCHES_1 = {
    "preds": [["the cat was under the bed"], ["the cat was found under the bed"]],
    "targets": [["the cat was found under the bed"], ["the tiny little cat was found under the big funny bed "]],
}


BATCHES_2 = {
    "preds": [["The quick brown fox jumps over the lazy dog"], ["My name is John"]],
    "targets": [["The quick brown dog jumps on the log."], ["Is your name John"]],
}


def _compute_rouge_score(preds: List[str], targets: List[str], use_stemmer: bool, rouge_level: str, metric: str):
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(targets, str):
        targets = [targets]
    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = BootstrapAggregator()
    for pred, target in zip(preds, targets):
        aggregator.add_scores(scorer.score(target, pred))
    rs_scores = aggregator.aggregate()
    rs_result = getattr(rs_scores[rouge_level].mid, metric)
    return rs_result


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
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(BATCHES_1["preds"], BATCHES_1["targets"]),
        pytest.param(BATCHES_2["preds"], BATCHES_2["targets"]),
    ],
)
class TestROUGEScore(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_rouge_score_class(self, ddp, dist_sync_on_step, preds, targets, pl_rouge_metric_key, use_stemmer):
        metric_args = {"use_stemmer": use_stemmer}

        rouge_level, metric = pl_rouge_metric_key.split("_")
        rouge_metric = partial(_compute_rouge_score, use_stemmer=use_stemmer, rouge_level=rouge_level, metric=metric)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=ROUGEScore,
            sk_metric=rouge_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.PREDS_FIRST,
            key=pl_rouge_metric_key,
        )

    def test_rouge_score_functional(self, preds, targets, pl_rouge_metric_key, use_stemmer):
        metric_args = {"use_stemmer": use_stemmer}

        rouge_level, metric = pl_rouge_metric_key.split("_")
        rouge_metric = partial(_compute_rouge_score, use_stemmer=use_stemmer, rouge_level=rouge_level, metric=metric)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=rouge_score,
            sk_metric=rouge_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.PREDS_FIRST,
            key=pl_rouge_metric_key,
        )


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
