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

import re
from functools import partial
from typing import Callable, Sequence, Union

import pytest
import torch
from torch import Tensor
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE
from typing_extensions import Literal

from unittests.text.helpers import TextTester, skip_on_connection_issues
from unittests.text.inputs import Input, _inputs_multiple_references, _inputs_single_sentence_single_reference

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import BootstrapAggregator
else:
    RougeScorer, BootstrapAggregator = object, object

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


# Some randomly adjusted input from CNN/DailyMail dataset which brakes the test
_preds = "A lawyer says him .\nMoschetto, 54 and prosecutors say .\nAuthority abc Moschetto  ."
_target = "A trainer said her and Moschetto, 54s or weapons say . \nAuthorities Moschetto of ."
_inputs_summarization = Input(preds=_preds, targets=_target)


def _compute_rouge_score(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[Union[str, Sequence[str]]]],
    use_stemmer: bool,
    rouge_level: str,
    metric: str,
    accumulate: Literal["avg", "best", None],
) -> Tensor:
    """Evaluate rouge scores from rouge-score package for baseline evaluation."""
    if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
        target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]

    if isinstance(preds, str) and accumulate:
        preds = [preds]

    if isinstance(target, str) and accumulate:
        target = [[target]]

    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    if not accumulate:
        rs_scores = scorer.score(target, preds)
        rs_result = getattr(rs_scores[rouge_level], metric)
        return torch.tensor(rs_result, dtype=torch.float)

    aggregator = BootstrapAggregator()

    for target_raw, pred_raw in zip(target, preds):
        list_results = [scorer.score(tgt, pred_raw) for tgt in target_raw]
        aggregator_avg = BootstrapAggregator()

        if accumulate == "best":
            key_curr = next(iter(list_results[0].keys()))
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
    return torch.tensor(rs_result, dtype=torch.float)


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
    """Test class for `ROUGEScore` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    @skip_on_connection_issues(reason="could not download nltk relevant data")
    def test_rouge_score_class(self, ddp, preds, targets, pl_rouge_metric_key, use_stemmer, accumulate):
        """Test class implementation of metric."""
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
            reference_metric=rouge_metric,
            metric_args=metric_args,
            key=pl_rouge_metric_key,
        )

    @skip_on_connection_issues(reason="could not download nltk relevant data")
    def test_rouge_score_functional(self, preds, targets, pl_rouge_metric_key, use_stemmer, accumulate):
        """Test functional implementation of metric."""
        metric_args = {"use_stemmer": use_stemmer, "accumulate": accumulate}

        rouge_level, metric = pl_rouge_metric_key.split("_")
        rouge_metric = partial(
            _compute_rouge_score, use_stemmer=use_stemmer, rouge_level=rouge_level, metric=metric, accumulate=accumulate
        )
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=rouge_score,
            reference_metric=rouge_metric,
            metric_args=metric_args,
            key=pl_rouge_metric_key,
        )


def test_rouge_metric_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    if not _NLTK_AVAILABLE:
        with pytest.raises(
            ModuleNotFoundError,
            match="ROUGE metric requires that `nltk` is installed."
            " Either as `pip install torchmetrics[text]` or `pip install nltk`.",
        ):
            ROUGEScore()


def test_rouge_metric_wrong_key_value_error():
    """Test errors are raised on wrongly provided keys."""
    key = ("rouge1", "rouge")

    with pytest.raises(ValueError, match="Got unknown rouge key rouge. Expected to be one of"):
        ROUGEScore(rouge_keys=key)

    with pytest.raises(ValueError, match="Got unknown rouge key rouge. Expected to be one of"):
        rouge_score(
            _inputs_single_sentence_single_reference.preds,
            _inputs_single_sentence_single_reference.targets,
            rouge_keys=key,
            accumulate="best",
        )


@pytest.mark.parametrize(
    "pl_rouge_metric_key",
    [
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_fmeasure",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_fmeasure",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_fmeasure",
        "rougeLsum_precision",
        "rougeLsum_recall",
        "rougeLsum_fmeasure",
    ],
)
@skip_on_connection_issues(reason="could not download nltk relevant data")
def test_rouge_metric_normalizer_tokenizer(pl_rouge_metric_key):
    """Test that rouge metric works for different rouge levels."""
    normalizer: Callable[[str], str] = lambda text: re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokenizer: Callable[[str], Sequence[str]] = lambda text: re.split(r"\s+", text)

    rouge_level, metric = pl_rouge_metric_key.split("_")
    original_score = _compute_rouge_score(
        preds=_inputs_single_sentence_single_reference.preds,
        target=_inputs_single_sentence_single_reference.targets,
        rouge_level=rouge_level,
        metric=metric,
        accumulate="best",
        use_stemmer=False,
    )

    scorer = ROUGEScore(
        normalizer=normalizer, tokenizer=tokenizer, rouge_keys=rouge_level, accumulate="best", use_stemmer=False
    )
    scorer.update(
        _inputs_single_sentence_single_reference.preds,
        _inputs_single_sentence_single_reference.targets,
    )
    metrics_score = scorer.compute()

    assert torch.isclose(metrics_score[rouge_level + "_" + metric], original_score)


@pytest.mark.parametrize(
    "pl_rouge_metric_key",
    [
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_fmeasure",
        "rougeLsum_precision",
        "rougeLsum_recall",
        "rougeLsum_fmeasure",
    ],
)
@pytest.mark.parametrize("use_stemmer", [False, True])
@skip_on_connection_issues(reason="could not download nltk relevant data")
def test_rouge_lsum_score(pl_rouge_metric_key, use_stemmer):
    """Specific tests to verify the correctness of Rouge-L and Rouge-LSum metric."""
    rouge_level, metric = pl_rouge_metric_key.split("_")
    original_score = _compute_rouge_score(
        preds=_inputs_summarization.preds,
        target=_inputs_summarization.targets,
        rouge_level=rouge_level,
        metric=metric,
        accumulate=None,
        use_stemmer=use_stemmer,
    )

    metrics_score = rouge_score(
        _inputs_summarization.preds,
        _inputs_summarization.targets,
        rouge_keys=rouge_level,
        use_stemmer=use_stemmer,
    )
    assert torch.isclose(metrics_score[rouge_level + "_" + metric], original_score)
