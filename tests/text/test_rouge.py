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
from typing import Sequence

import pytest
import torch

from tests.text.helpers import TextTester
from tests.text.inputs import _inputs_multiple_references, _inputs_single_sentence_single_reference
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import BootstrapAggregator
else:
    RougeScorer, BootstrapAggregator = object, object

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def _compute_rouge_score(
    preds: Sequence[str],
    targets: Sequence[Sequence[str]],
    use_stemmer: bool,
    rouge_level: str,
    metric: str,
    accumulate: str,
):
    """Evaluates rouge scores from rouge-score package for baseline evaluation."""
    if isinstance(targets, list) and all(isinstance(target, str) for target in targets):
        targets = [targets] if isinstance(preds, str) else [[target] for target in targets]

    if isinstance(preds, str):
        preds = [preds]

    if isinstance(targets, str):
        targets = [[targets]]

    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = BootstrapAggregator()

    for target_raw, pred_raw in zip(targets, preds):
        list_results = [scorer.score(target, pred_raw) for target in target_raw]
        aggregator_avg = BootstrapAggregator()

        if accumulate == "best":
            key_curr = list(list_results[0].keys())[0]
            all_fmeasure = torch.tensor([v[key_curr].fmeasure for v in list_results])
            highest_idx = torch.argmax(all_fmeasure).item()
            aggregator.add_scores(list_results[highest_idx])
        elif accumulate == "avg":
            for _score in list_results:
                aggregator_avg.add_scores(_score)
            _score = {rouge_key: scores.mid for rouge_key, scores in aggregator_avg.aggregate().items()}
            aggregator.add_scores(_score)
        else:
            raise ValueError(f"Got unknown accumulate value {accumulate}. Expected to be one of ['best', 'avg']")

    rs_scores = aggregator.aggregate()
    rs_result = getattr(rs_scores[rouge_level].mid, metric)
    return rs_result


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer"],
    [
        ("rouge1_precision", True),
        ("rouge1_recall", True),
        ("rouge1_fmeasure", False),
        ("rouge2_precision", False),
        ("rouge2_recall", True),
        ("rouge2_fmeasure", True),
        ("rougeL_precision", False),
        ("rougeL_recall", False),
        ("rougeL_fmeasure", True),
        ("rougeLsum_precision", True),
        ("rougeLsum_recall", False),
        ("rougeLsum_fmeasure", False),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_multiple_references.preds, _inputs_multiple_references.targets),
    ],
)
@pytest.mark.parametrize("accumulate", ["avg", "best"])
class TestROUGEScore(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_rouge_score_class(
        self, ddp, dist_sync_on_step, preds, targets, pl_rouge_metric_key, use_stemmer, accumulate
    ):
        metric_args = {"use_stemmer": use_stemmer, "accumulate": accumulate}
        rouge_level, metric = pl_rouge_metric_key.split("_")
        rouge_metric = partial(
            _compute_rouge_score, use_stemmer=use_stemmer, rouge_level=rouge_level, metric=metric, accumulate=accumulate
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=ROUGEScore,
            sk_metric=rouge_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            key=pl_rouge_metric_key,
        )

    def test_rouge_score_functional(self, preds, targets, pl_rouge_metric_key, use_stemmer, accumulate):
        metric_args = {"use_stemmer": use_stemmer, "accumulate": accumulate}

        rouge_level, metric = pl_rouge_metric_key.split("_")
        rouge_metric = partial(
            _compute_rouge_score, use_stemmer=use_stemmer, rouge_level=rouge_level, metric=metric, accumulate=accumulate
        )
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=rouge_score,
            sk_metric=rouge_metric,
            metric_args=metric_args,
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
        rouge_score(
            _inputs_single_sentence_single_reference.preds,
            _inputs_single_sentence_single_reference.targets,
            rouge_keys=key,
            accumulate="best",
        )
