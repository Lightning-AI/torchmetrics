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
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import AggregateScore, BootstrapAggregator
else:
    RougeScorer, AggregateScore, BootstrapAggregator = object, object, object

ALLOWED_ROUGE_KEYS = (
    "rouge1",
    "rouge2",
    "rouge3",
    "rouge4",
    "rouge5",
    "rouge6",
    "rouge7",
    "rouge8",
    "rouge9",
    "rougeL",
    "rougeLsum",
)


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    if _NLTK_AVAILABLE:
        import nltk

        nltk.download("punkt", quiet=True, force=False)

    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def format_rouge_results(result: Dict[str, AggregateScore], decimal_places: int = 4) -> Dict[str, Tensor]:
    """Formats the computed (aggregated) rouge score to a dictionary of tensors format."""
    flattened_result = {}
    for rouge_key, rouge_aggregate_score in result.items():
        for stat in ["precision", "recall", "fmeasure"]:
            mid = rouge_aggregate_score.mid
            score = round(getattr(mid, stat), decimal_places)
            flattened_result[f"{rouge_key}_{stat}"] = tensor(score, dtype=torch.float)
    return flattened_result


def _rouge_score_update(
    preds: List[str],
    targets: List[str],
    scorer: RougeScorer,
    aggregator: BootstrapAggregator,
    newline_sep: bool = False,
) -> None:
    """Update the rouge score with the current set of predicted and target sentences.

    Args:
        preds:
            An iterable of predicted sentences.
        targets:
            An iterable of target sentences.
        scorer:
            An instance of the ``RougeScorer`` class from the ``rouge_score`` package.
        aggregator:
            An instance of the ``BootstrapAggregator`` from the from the ``rouge_score`` package.
        newline_sep:
            New line separate the inputs.

    Example:
        >>> targets = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> aggregator = BootstrapAggregator()
        >>> scorer = RougeScorer(rouge_types=("rouge1", "rouge2", "rougeL", "rougeLsum"), use_stemmer=False)
        >>> _rouge_score_update(preds, targets, scorer=scorer, aggregator=aggregator, newline_sep=False)
    """
    for pred, target in zip(preds, targets):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            target = add_newline_to_end_of_each_sentence(target)
        results = scorer.score(pred, target)
        aggregator.add_scores(results)


def _rouge_score_compute(aggregator: BootstrapAggregator, decimal_places: int = 4) -> Dict[str, Tensor]:
    """Compute the combined ROUGE metric for all the input set of predicted and target sentences.

    Args:
        aggregator:
            An instance of the ``BootstrapAggregator`` from the from the ``rouge_score`` package.
        decimal_places:
            The number of digits to round the computed the values to.
    """
    result = aggregator.aggregate()
    return format_rouge_results(result, decimal_places)


def rouge_score(
    preds: Union[str, List[str]],
    targets: Union[str, List[str]],
    newline_sep: bool = False,
    use_stemmer: bool = False,
    rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
    decimal_places: int = 4,
) -> Dict[str, Tensor]:
    """Calculate `ROUGE score <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_, used for automatic summarization.

    Args:
        preds:
            An iterable of predicted sentences.
        targets:
            An iterable of target sentences.
        newline_sep:
            New line separate the inputs.
        use_stemmer:
            Use Porter stemmer to strip word suffixes to improve matching.
        rouge_keys:
            A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        decimal_places:
            The number of digits to round the computed the values to.

    Return:
        Python dictionary of rouge scores for each input rouge key.

    Example:
        >>> targets = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> from pprint import pprint
        >>> pprint(rouge_score(preds, targets))  # doctest: +NORMALIZE_WHITESPACE +SKIP
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

    Raises:
        ValueError:
            If the python packages ``nltk`` or ``rouge-score`` are not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    if not (_NLTK_AVAILABLE and _ROUGE_SCORE_AVAILABLE):
        raise ValueError(
            "ROUGE metric requires that both nltk and rouge-score is installed."
            " Either as `pip install torchmetrics[text]` or `pip install nltk rouge-score`"
        )

    if not isinstance(rouge_keys, tuple):
        rouge_keys = tuple([rouge_keys])
    for key in rouge_keys:
        if key not in ALLOWED_ROUGE_KEYS:
            raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")

    if isinstance(preds, str):
        preds = [preds]

    if isinstance(targets, str):
        targets = [targets]

    aggregator = BootstrapAggregator()
    scorer = RougeScorer(rouge_keys, use_stemmer=use_stemmer)

    _rouge_score_update(preds, targets, scorer=scorer, aggregator=aggregator, newline_sep=newline_sep)
    return _rouge_score_compute(aggregator=aggregator, decimal_places=decimal_places)
