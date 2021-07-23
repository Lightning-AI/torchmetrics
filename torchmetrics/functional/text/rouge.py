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
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, tensor

from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score import rouge_scorer
    from rouge_score.scoring import AggregateScore, BootstrapAggregator, Score
else:
    AggregateScore, Score, BootstrapAggregator = None, None, object

nltk = None
if _NLTK_AVAILABLE:
    import nltk
    nltk.download("punkt", quiet=True)


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def format_rouge_results(result: Dict[str, AggregateScore], decimal_places: int = 4) -> Dict[str, Tensor]:
    flattened_result = {}
    for rouge_key, rouge_aggregate_score in result.items():
        for stat in ["precision", "recall", "fmeasure"]:
            mid = rouge_aggregate_score.mid
            score = round(getattr(mid, stat), decimal_places)
            flattened_result[f"{rouge_key}_{stat}"] = tensor(score)
    return flattened_result


class RougeBatchAggregator(BootstrapAggregator):
    """
    Aggregates rouge scores and provides confidence intervals.
    """

    def aggregate(self):
        """
        Override function to wrap the final results in `Score` objects.
        This is due to the scores being replaced with a list of torch tensors.
        """
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple((Score(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def add_scores(self, scores):
        self._scores = scores


def _rouge_score_update(
    pred_lns: List[str],
    tgt_lns: List[str],
    scores: Dict[str, List[Tensor]],
    scorer: rouge_scorer.RougeScorer,
    rouge_newline_sep: bool = False,
) -> None:

    for pred, tgt in zip(pred_lns, tgt_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if rouge_newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        results = scorer.score(pred, tgt)
        for key, score in results.items():
            score = tensor([score.precision, score.recall, score.fmeasure])
            scores[key].append(score)


def _rouge_score_compute(scores: Dict[str, List[Tensor]], aggregator: RougeBatchAggregator) -> Dict[str, Tensor]:
    aggregator.add_scores(scores)
    result = aggregator.aggregate()
    return format_rouge_results(result)


def rouge_score(
    pred_lns: List[str],
    tgt_lns: List[str],
    rouge_newline_sep: bool = False,
    use_stemmer: bool = False,
    rouge_keys: Tuple[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum")
) -> Dict[str, Tensor]:
    """
    Calculate `ROUGE score <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_.
    Used for automatic summarization.

    Args:
        pred_lns:
            An iterable of
        tgt_lns:
            An iterable of
        rouge_newline_sep:

        use_stemmer:

        rouge_keys

    Return:
        Python dictionary of rouge scores for each input rouge key.

    Example:
        >>> target = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> from pprint import pprint
        >>> pprint(rouge_score(preds, target))  # doctest: +NORMALIZE_WHITESPACE +SKIP
        {'rouge1_fmeasure': 0.25,
         'rouge1_precision': 0.25,
         'rouge1_recall': 0.25,
         'rouge2_fmeasure': 0.0,
         'rouge2_precision': 0.0,
         'rouge2_recall': 0.0,
         'rougeL_fmeasure': 0.25,
         'rougeL_precision': 0.25,
         'rougeL_recall': 0.25,
         'rougeLsum_fmeasure': 0.25,
         'rougeLsum_precision': 0.25,
         'rougeLsum_recall': 0.25}

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    aggregator = RougeBatchAggregator()
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    scores = {key: [] for key in rouge_keys}

    _rouge_score_update(pred_lns, tgt_lns, scores=scores, scorer=scorer, rouge_newline_sep=rouge_newline_sep)
    return _rouge_score_compute(scores, aggregator=aggregator)
