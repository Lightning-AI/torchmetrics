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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.utilities.imports import _NLTK_AVAILABLE

if _NLTK_AVAILABLE:
    import nltk

ALLOWED_ROUGE_KEYS = {
    "rouge1": 1,
    "rouge2": 2,
    "rouge3": 3,
    "rouge4": 4,
    "rouge5": 5,
    "rouge6": 6,
    "rouge7": 7,
    "rouge8": 8,
    "rouge9": 9,
    "rougeL": "L",
    "rougeLsum": "Lsum",
}


@dataclass
class _RougeScore:
    precision: float = 0.0
    recall: float = 0.0
    fmeasure: float = 0.0


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    if _NLTK_AVAILABLE:
        nltk.download("punkt", quiet=True, force=False)

    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def _compute_metrics(hits_or_lcs: int, pred_len: int, target_len: int) -> _RougeScore:
    """This computes precision, recall and F1 score based on hits/lcs, and the length lists of predicted and target
    sentences.

    Args:
        hits_or_lcs:
            A number of matches or a length of the longest common subsequence
        pred_len:
            A length of a tokenized predicted sentence.
        target_len:
            A length of a tokenized target sentence.
    """
    precision = hits_or_lcs / pred_len
    recall = hits_or_lcs / target_len
    if precision == recall == 0.0:
        return _RougeScore()

    fmeasure = 2 * precision * recall / (precision + recall)
    return _RougeScore(precision, recall, fmeasure)


def _lcs(pred_tokens: List[str], target_tokens: List[str]) -> int:
    """General DP algorithm to compute the length of the longest common subsequence."""
    LCS = [[0] * (len(pred_tokens) + 1) for _ in range(len(target_tokens) + 1)]
    for i in range(1, len(target_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if target_tokens[i - 1] == pred_tokens[j - 1]:
                LCS[i][j] = LCS[i - 1][j - 1] + 1
            else:
                LCS[i][j] = max(LCS[i - 1][j], LCS[i][j - 1])
    return LCS[-1][-1]


def _normalize_text(text: str, stemmer: 'nltk.stem.porter.PorterStemmer') -> str:
    """Rouge score should be calculated only over lowercased words and digits. Optionally, Pprter stemmer can be used 
    to strip word suffixes to improve matching.

        Args:
            text:
                An input sentence
            use_stemmer:
                Use Porter stemmer to strip word suffixes to improve matching.
            
    """
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    if stemmer:
        # Only stem words more than 3 characters long.
        text = " ".join(stemmer.stem(x) if len(x) > 3 else x for x in text.split())
    return text


def _rouge_n_score(pred: str, target: str, n_gram: int) -> _RougeScore:
    """This computes precision, recall and F1 score for the Rouge-N metric.

    Args:
        pred:
            A predicted sentence.
        target:
            A target sentence.
        n_gram:
            N-gram overlap.
    """
    pred_tokenized, target_tokenized = _tokenize(pred, n_gram), _tokenize(target, n_gram)
    pred_len, target_len = len(pred_tokenized), len(target_tokenized)
    if pred_len == 0 or target_len == 0:
        return _RougeScore()

    pred_counter, target_counter = defaultdict(int), defaultdict(int)
    for w in pred_tokenized:
        pred_counter[w] += 1
    for w in target_tokenized:
        target_counter[w] += 1
    hits = sum(min(pred_counter[w], target_counter[w]) for w in set(pred_tokenized))
    return _compute_metrics(hits, pred_len, target_len)


def _rouge_l_score(pred: str, target: str, summary_level: bool = False) -> _RougeScore:
    """This computes precision, recall and F1 score for the Rouge-L or Rouge-LSum metric.

    Args:
        pred:
            A predicted sentence.
        target:
            A target sentence.
        summary_level:
            Calculate summary-level metric.
    """
    if summary_level:
        pass
    pred_tokenized, target_tokenized = _tokenize(pred, 1), _tokenize(target, 1)
    pred_len, target_len = len(pred_tokenized), len(target_tokenized)
    if pred_len == 0 or target_len == 0:
        return _RougeScore()

    lcs = _lcs(pred_tokenized, target_tokenized)
    return _compute_metrics(lcs, pred_len, target_len)


def _rouge_score_update(
    preds: List[str],
    targets: List[str],
    rouge_keys_values: Tuple[Union[int, str], ...],
    stemmer: bool = False,
    newline_sep: bool = False,
) -> Dict[Union[int, str], List[_RougeScore]]:
    """Update the rouge score with the current set of predicted and target sentences.

    Args:
        preds:
            An iterable of predicted sentences.
        targets:
            An iterable of target sentences.
        rouge_keys_values:
            List of N-grams/'L'/'Lsum' arguments.
        use_stemmer:
            Use Porter stemmer to strip word suffixes to improve matching.
        newline_sep:
            New line separate the inputs.

    Example:
        >>> targets = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> _rouge_score_update(preds, targets, rouge_keys_values=[1, 2, 3, 'L'])
    """
    results: Dict[Union[int, str], List[_RougeScore]] = {rouge_key: [] for rouge_key in rouge_keys_values}
    for pred, target in zip(preds, targets):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            target = add_newline_to_end_of_each_sentence(target)
        pred, target = _normalize_text(pred, stemmer), _normalize_text(target, stemmer)

        for rouge_key in rouge_keys_values:
            results[rouge_key].append(
                _rouge_n_score(pred, target, rouge_key) if isinstance(rouge_key, int)
                else _rouge_l_score(pred, target, summary_level=True if "Lsum" else False)
            )
    return results


def _rouge_score_compute(sentence_results: Dict[Union[int, str], List[_RougeScore]]) -> Dict[str, Tensor]:
    """Compute the combined ROUGE metric for all the input set of predicted and target sentences.

    Args:
        sentence_results:
            Rouge-N/Rouge-L/Rouge-LSum metrics calculated for single sentence. 
    """
    results = {}
    for rouge_key, scores in sentence_results.items():
        res = torch.tensor(
            [(score.precision, score.recall, score.fmeasure) for score in scores]
        ).mean(0)
        results[f"rouge{rouge_key}_precision"] = res[0]
        results[f"rouge{rouge_key}_recall"] = res[1]
        results[f"rouge{rouge_key}_fmeasure"] = res[2]

    return results


def rouge_score(
    preds: Union[str, List[str]],
    targets: Union[str, List[str]],
    newline_sep: bool = False,
    use_stemmer: bool = False,
    rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
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
            If the python package ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    if not (_NLTK_AVAILABLE):
        raise ValueError(
            "ROUGE metric requires that nltk is installed."
            " Either as `pip install torchmetrics[text]` or `pip install nltk rouge-score`"
        )
    stemmer = nltk.stem.porter.PorterStemmer() if _NLTK_AVAILABLE and use_stemmer else None

    if not isinstance(rouge_keys, tuple):
        rouge_keys = tuple([rouge_keys])
    for key in rouge_keys:
        if key not in ALLOWED_ROUGE_KEYS.keys():
            raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {list(ALLOWED_ROUGE_KEYS.keys())}")
    rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]

    if isinstance(preds, str):
        preds = [preds]

    if isinstance(targets, str):
        targets = [targets]

    sentence_results = _rouge_score_update(preds, targets, rouge_keys_values, stemmer, newline_sep)
    return _rouge_score_compute(sentence_results)


def _tokenize(text: str, n_gram: int) -> List[str]:
    """Retun the list of N-grams from the input text.
    
    Args:
        text:
            An input sentence.
        n_gram
            N-gram size to return.
    """
    tokens = text.split()
    n_grams_list = [' '.join(tokens[i:i + n_gram]) for i in range(len(tokens) - n_gram + 1)]
    return n_grams_list
