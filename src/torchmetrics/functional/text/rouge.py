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
import urllib.request
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import HTTPError

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _NLTK_AVAILABLE

__doctest_requires__ = {("rouge_score", "_rouge_score_update"): ["nltk"]}

ALLOWED_ROUGE_KEYS: Dict[str, Union[int, str]] = {
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
ALLOWED_ACCUMULATE_VALUES = ("avg", "best")


def _ensure_nltk_punkt_is_downloaded() -> None:
    """Check whether `nltk` `punkt` is downloaded.

    If not, try to download if a machine is connected to the internet.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True, force=False, halt_on_error=False, raise_on_error=True)
        except ValueError as err:
            raise OSError(
                "`nltk` resource `punkt` is not available on a disk and cannot be downloaded as a machine is not "
                "connected to the internet."
            ) from err


def _split_sentence(x: str) -> Sequence[str]:
    """Split sentence to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    if not _NLTK_AVAILABLE:
        raise ModuleNotFoundError("ROUGE-Lsum calculation requires that `nltk` is installed. Use `pip install nltk`.")
    import nltk

    _ensure_nltk_punkt_is_downloaded()

    re.sub("<n>", "", x)  # remove pegasus newline char
    return nltk.sent_tokenize(x)


def _compute_metrics(hits_or_lcs: int, pred_len: int, target_len: int) -> Dict[str, Tensor]:
    """Compute overall metrics.

    This function computes precision, recall and F1 score based on hits/lcs, the length of lists of tokenizer
    predicted and target sentences.

    Args:
        hits_or_lcs: A number of matches or a length of the longest common subsequence.
        pred_len: A length of a tokenized predicted sentence.
        target_len: A length of a tokenized target sentence.
    """
    precision = hits_or_lcs / pred_len
    recall = hits_or_lcs / target_len
    if precision == recall == 0.0:
        return {"precision": tensor(0.0), "recall": tensor(0.0), "fmeasure": tensor(0.0)}

    fmeasure = 2 * precision * recall / (precision + recall)
    return {"precision": tensor(precision), "recall": tensor(recall), "fmeasure": tensor(fmeasure)}


def _lcs(
    pred_tokens: Sequence[str], target_tokens: Sequence[str], return_full_table: bool = False
) -> Union[int, Sequence[Sequence[int]]]:
    """DP algorithm to compute the length of the longest common subsequence.

    Args:
        pred_tokens: A tokenized predicted sentence.
        target_tokens: A tokenized target sentence.
        return_full_table: If the full table of logest common subsequence should be returned or just the largest
    """
    lcs = [[0] * (len(pred_tokens) + 1) for _ in range(len(target_tokens) + 1)]
    for i in range(1, len(target_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if target_tokens[i - 1] == pred_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    if return_full_table:
        return lcs
    return lcs[-1][-1]


def _backtracked_lcs(
    lcs_table: Sequence[Sequence[int]], pred_tokens: Sequence[str], target_tokens: Sequence[str]
) -> Sequence[int]:
    """Backtrack LCS table.

    Args:
        lcs_table: A table containing information for the calculation of the longest common subsequence.
        pred_tokens: A tokenized predicted sentence.
        target_tokens: A tokenized target sentence.
    """
    i = len(pred_tokens)
    j = len(target_tokens)
    backtracked_lcs: List[int] = []
    while i > 0 and j > 0:
        if pred_tokens[i - 1] == target_tokens[j - 1]:
            backtracked_lcs.insert(0, j - 1)
            i -= 1
            j -= 1
        elif lcs_table[j][i - 1] > lcs_table[j - 1][i]:
            i -= 1
        else:
            j -= 1
    return backtracked_lcs


def _union_lcs(pred_tokens_list: Sequence[Sequence[str]], target_tokens: Sequence[str]) -> Sequence[str]:
    r"""Find union LCS between a target sentence and iterable of predicted tokens.

    Args:
        pred_tokens_list: A tokenized predicted sentence split by ``'\n'``.
        target_tokens: A tokenized single part of target sentence split by ``'\n'``.
    """

    def lcs_ind(pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> Sequence[int]:
        """Return one of the longest of longest common subsequence via backtracked lcs table."""
        lcs_table: Sequence[Sequence[int]] = _lcs(pred_tokens, target_tokens, return_full_table=True)  # type: ignore
        return _backtracked_lcs(lcs_table, pred_tokens, target_tokens)

    def find_union(lcs_tables: Sequence[Sequence[int]]) -> Sequence[int]:
        """Find union LCS given a list of LCS."""
        return sorted(set().union(*lcs_tables))

    lcs_tables = [lcs_ind(pred_tokens, target_tokens) for pred_tokens in pred_tokens_list]
    return [target_tokens[i] for i in find_union(lcs_tables)]


def _normalize_and_tokenize_text(
    text: str,
    stemmer: Optional[Any] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
) -> Sequence[str]:
    """Rouge score should be calculated only over lowercased words and digits.

    Optionally, Porter stemmer can be used to strip word suffixes to improve matching. The text normalization follows
    the implemantion from `Rouge score_Text Normalizition`_.

    Args:
        text: An input sentence.
        stemmer: Porter stemmer instance to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, splitting by spaces is default
            This function must take a ``str`` and return ``Sequence[str]``
    """
    # If normalizer is none, replace any non-alpha-numeric characters with spaces.
    text = normalizer(text) if callable(normalizer) else re.sub(r"[^a-z0-9]+", " ", text.lower())

    # If tokenizer is none, spliting by spaces
    tokens = tokenizer(text) if callable(tokenizer) else re.split(r"\s+", text)

    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]

    # One final check to drop any empty or invalid tokens.
    return [x for x in tokens if (isinstance(x, str) and len(x) > 0)]


def _rouge_n_score(pred: Sequence[str], target: Sequence[str], n_gram: int) -> Dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-N metric.

    Args:
        pred: A predicted sentence.
        target: A target sentence.
        n_gram: N-gram overlap.
    """

    def _create_ngrams(tokens: Sequence[str], n: int) -> Counter:
        ngrams: Counter = Counter()
        for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    pred_ngrams, target_ngrams = _create_ngrams(pred, n_gram), _create_ngrams(target, n_gram)
    pred_len, target_len = sum(pred_ngrams.values()), sum(target_ngrams.values())
    if 0 in (pred_len, target_len):
        return {"precision": tensor(0.0), "recall": tensor(0.0), "fmeasure": tensor(0.0)}

    # It is sufficient to take a set(pred_tokenized) for hits count as we consider intersenction of pred & target
    hits = sum(min(pred_ngrams[w], target_ngrams[w]) for w in set(pred_ngrams))
    return _compute_metrics(hits, max(pred_len, 1), max(target_len, 1))


def _rouge_l_score(pred: Sequence[str], target: Sequence[str]) -> Dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-L metric.

    Args:
        pred: A predicted sentence.
        target: A target sentence.
    """
    pred_len, target_len = len(pred), len(target)
    if 0 in (pred_len, target_len):
        return {"precision": tensor(0.0), "recall": tensor(0.0), "fmeasure": tensor(0.0)}

    lcs: int = _lcs(pred, target)  # type: ignore
    return _compute_metrics(lcs, pred_len, target_len)


def _rouge_lsum_score(pred: Sequence[Sequence[str]], target: Sequence[Sequence[str]]) -> Dict[str, Tensor]:
    r"""Compute precision, recall and F1 score for the Rouge-LSum metric.

    More information can be found in Section 3.2 of the referenced paper [1]. This implementation follow the official
    implementation from:
    https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py.

    Args:
        pred: An iterable of predicted sentence split by '\n'.
        target: An iterable target sentence split by '\n'.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin. https://aclanthology.org/W04-1013/
    """
    pred_len = sum(map(len, pred))
    target_len = sum(map(len, target))
    if 0 in (pred_len, target_len):
        return {"precision": tensor(0.0), "recall": tensor(0.0), "fmeasure": tensor(0.0)}

    # Get token counts
    def _get_token_counts(sentences: Sequence[Sequence[str]]) -> Counter:
        ngrams: Counter = Counter()
        for sentence in sentences:
            ngrams.update(sentence)
        return ngrams

    pred_tokens_count = _get_token_counts(pred)
    target_tokens_count = _get_token_counts(target)

    # Calculate hits
    hits = 0
    for tgt in target:
        lcs = _union_lcs(pred, tgt)
        for token in lcs:
            if pred_tokens_count[token] > 0 and target_tokens_count[token] > 0:
                hits += 1
                pred_tokens_count[token] -= 1
                target_tokens_count[token] -= 1

    return _compute_metrics(hits, pred_len, target_len)


def _rouge_score_update(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    rouge_keys_values: List[Union[int, str]],
    accumulate: str,
    stemmer: Optional[Any] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
) -> Dict[Union[int, str], List[Dict[str, Tensor]]]:
    """Update the rouge score with the current set of predicted and target sentences.

    Args:
        preds: An iterable of predicted sentences.
        target: An iterable of iterable of target sentences.
        rouge_keys_values: List of N-grams/'L'/'Lsum' arguments.
        accumulate: Useful incase of multi-reference rouge score.
            ``avg`` takes the avg of all references with respect to predictions
            ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.
            Allowed values are ``avg`` and ``best``.
        stemmer: Porter stemmer instance to strip word suffixes to improve matching.
        normalizer:
            A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a `str` and return a `str`.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, spliting by spaces is default
            This function must take a `str` and return `Sequence[str]`

    Example:
        >>> preds = "My name is John".split()
        >>> target = "Is your name John".split()
        >>> from pprint import pprint
        >>> score = _rouge_score_update(preds, target, rouge_keys_values=[1, 2, 3, 'L'], accumulate='best')
        >>> pprint(score)
        {1: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         2: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         3: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         'L': [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}]}
    """
    results: Dict[Union[int, str], List[Dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}

    for pred_raw, target_raw in zip(preds, target):
        result_inner: Dict[Union[int, str], Dict[str, Tensor]] = {rouge_key: {} for rouge_key in rouge_keys_values}
        result_avg: Dict[Union[int, str], List[Dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}
        list_results = []
        pred = _normalize_and_tokenize_text(pred_raw, stemmer, normalizer, tokenizer)
        if "Lsum" in rouge_keys_values:
            pred_lsum = [
                _normalize_and_tokenize_text(pred_sentence, stemmer, normalizer, tokenizer)
                for pred_sentence in _split_sentence(pred_raw)
            ]

        for target_raw_inner in target_raw:
            tgt = _normalize_and_tokenize_text(target_raw_inner, stemmer, normalizer, tokenizer)

            if "Lsum" in rouge_keys_values:
                target_lsum = [
                    _normalize_and_tokenize_text(tgt_sentence, stemmer, normalizer, tokenizer)
                    for tgt_sentence in _split_sentence(target_raw_inner)
                ]

            for rouge_key in rouge_keys_values:
                if isinstance(rouge_key, int):
                    score = _rouge_n_score(pred, tgt, rouge_key)
                elif rouge_key == "L":
                    score = _rouge_l_score(pred, tgt)
                elif rouge_key == "Lsum":
                    score = _rouge_lsum_score(pred_lsum, target_lsum)
                result_inner[rouge_key] = score
                result_avg[rouge_key].append(score)
            list_results.append(result_inner.copy())

        if accumulate == "best":
            key_curr = rouge_keys_values[0]
            all_fmeasure = torch.tensor([v[key_curr]["fmeasure"] for v in list_results])
            highest_idx = int(torch.argmax(all_fmeasure).item())

            for rouge_key in rouge_keys_values:
                results[rouge_key].append(list_results[highest_idx][rouge_key])  # noqa: PERF401 # todo

        elif accumulate == "avg":
            new_result_avg: Dict[Union[int, str], Dict[str, Tensor]] = {
                rouge_key: {} for rouge_key in rouge_keys_values
            }
            for rouge_key, metrics in result_avg.items():
                _dict_metric_score_batch: Dict[str, List[Tensor]] = {}
                for metric in metrics:
                    for _type, value in metric.items():
                        if _type not in _dict_metric_score_batch:
                            _dict_metric_score_batch[_type] = []
                        _dict_metric_score_batch[_type].append(value)

                new_result_avg[rouge_key] = {
                    _type: torch.tensor(_dict_metric_score_batch[_type]).mean() for _type in _dict_metric_score_batch
                }

            for rouge_key in rouge_keys_values:
                results[rouge_key].append(new_result_avg[rouge_key])  # noqa: PERF401 # todo

    return results


def _rouge_score_compute(sentence_results: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
    """Compute the combined ROUGE metric for all the input set of predicted and target sentences.

    Args:
        sentence_results: Rouge-N/Rouge-L/Rouge-LSum metrics calculated for single sentence.
    """
    results: Dict[str, Tensor] = {}
    # Obtain mean scores for individual rouge metrics
    if sentence_results == {}:
        return results

    for rouge_key, scores in sentence_results.items():
        results[rouge_key] = torch.tensor(scores).mean()

    return results


def rouge_score(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str], Sequence[Sequence[str]]],
    accumulate: Literal["avg", "best"] = "best",
    use_stemmer: bool = False,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
    rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
) -> Dict[str, Tensor]:
    """Calculate `Calculate Rouge Score`_ , used for automatic summarization.

    Args:
        preds: An iterable of predicted sentences or a single predicted sentence.
        target:
            An iterable of iterables of target sentences or an iterable of target sentences or a single target sentence.
        accumulate:
            Useful incase of multi-reference rouge score.

            - ``avg`` takes the avg of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.

        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer: A user's own tokenizer function. If this is ``None``, spliting by spaces is default
            This function must take a ``str`` and return ``Sequence[str]``
        rouge_keys: A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.

    Return:
        Python dictionary of rouge scores for each input rouge key.

    Example:
        >>> from torchmetrics.functional.text.rouge import rouge_score
        >>> preds = "My name is John"
        >>> target = "Is your name John"
        >>> from pprint import pprint
        >>> pprint(rouge_score(preds, target))
        {'rouge1_fmeasure': tensor(0.7500),
         'rouge1_precision': tensor(0.7500),
         'rouge1_recall': tensor(0.7500),
         'rouge2_fmeasure': tensor(0.),
         'rouge2_precision': tensor(0.),
         'rouge2_recall': tensor(0.),
         'rougeL_fmeasure': tensor(0.5000),
         'rougeL_precision': tensor(0.5000),
         'rougeL_recall': tensor(0.5000),
         'rougeLsum_fmeasure': tensor(0.5000),
         'rougeLsum_precision': tensor(0.5000),
         'rougeLsum_recall': tensor(0.5000)}


    Raises:
        ModuleNotFoundError:
            If the python package ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin. https://aclanthology.org/W04-1013/
    """
    if use_stemmer:
        if not _NLTK_AVAILABLE:
            raise ModuleNotFoundError("Stemmer requires that `nltk` is installed. Use `pip install nltk`.")
        import nltk

    stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None

    if not isinstance(rouge_keys, tuple):
        rouge_keys = (rouge_keys,)
    for key in rouge_keys:
        if key not in ALLOWED_ROUGE_KEYS.keys():
            raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {list(ALLOWED_ROUGE_KEYS.keys())}")
    rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]

    if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
        target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]

    if isinstance(preds, str):
        preds = [preds]

    if isinstance(target, str):
        target = [[target]]

    sentence_results: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(
        preds,
        target,
        rouge_keys_values,
        stemmer=stemmer,
        normalizer=normalizer,
        tokenizer=tokenizer,
        accumulate=accumulate,
    )

    output: Dict[str, List[Tensor]] = {
        f"rouge{rouge_key}_{tp}": [] for rouge_key in rouge_keys_values for tp in ["fmeasure", "precision", "recall"]
    }
    for rouge_key, metrics in sentence_results.items():
        for metric in metrics:
            for tp, value in metric.items():
                output[f"rouge{rouge_key}_{tp}"].append(value)  # noqa: PERF401 # todo

    return _rouge_score_compute(output)
