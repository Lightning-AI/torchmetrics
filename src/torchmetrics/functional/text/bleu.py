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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors and @sluks
# Date: 2020-07-18
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#bleu_score
from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor


def _count_ngram(ngram_input_list: Sequence[str], n_gram: int) -> Counter:
    """Count how many times each word appears in a given text with ngram.

    Args:
        ngram_input_list: A list of translated text or reference texts
        n_gram: gram value ranged 1 to 4

    Return:
        ngram_counter: a collections.Counter object of ngram
    """
    ngram_counter: Counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(ngram_input_list) - i + 1):
            ngram_key = tuple(ngram_input_list[j : (i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter


def _tokenize_fn(sentence: str) -> Sequence[str]:
    """Tokenizes sentence into list of words.

    Args:
        sentence: A sentence separated by white space.

    Return:
        List of words
    """
    return sentence.split()


def _bleu_score_update(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    numerator: Tensor,
    denominator: Tensor,
    preds_len: Tensor,
    target_len: Tensor,
    n_gram: int = 4,
    tokenizer: Callable[[str], Sequence[str]] = _tokenize_fn,
) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute the BLEU score.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        preds_len: count of words in a candidate prediction
        target_len: count of words in a reference translation
        target: count of words in a reference translation
        n_gram: gram value ranged 1 to 4
        tokenizer: A function that turns sentence into list of words
    """
    target_: Sequence[Sequence[Sequence[str]]] = [[tokenizer(line) if line else [] for line in t] for t in target]
    preds_: Sequence[Sequence[str]] = [tokenizer(line) if line else [] for line in preds]

    for pred, targets in zip(preds_, target_):
        preds_len += len(pred)
        target_len_list = [len(tgt) for tgt in targets]
        target_len_diff = [abs(len(pred) - x) for x in target_len_list]
        target_len += target_len_list[target_len_diff.index(min(target_len_diff))]
        preds_counter: Counter = _count_ngram(pred, n_gram)
        target_counter: Counter = Counter()

        for tgt in targets:
            target_counter |= _count_ngram(tgt, n_gram)

        ngram_counter_clip = preds_counter & target_counter

        for counter_clip in ngram_counter_clip:
            numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

        for counter in preds_counter:
            denominator[len(counter) - 1] += preds_counter[counter]

    return preds_len, target_len


def _bleu_score_compute(
    preds_len: Tensor,
    target_len: Tensor,
    numerator: Tensor,
    denominator: Tensor,
    n_gram: int,
    weights: Sequence[float],
    smooth: bool,
) -> Tensor:
    """Compute the BLEU score.

    Args:
        preds_len: count of words in a candidate translation
        target_len: count of words in a reference translation
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        n_gram: gram value ranged 1 to 4
        weights: Weights used for unigrams, bigrams, etc. to calculate BLEU score.
        smooth: Whether to apply smoothing
    """
    device = numerator.device
    if min(numerator) == 0.0:
        return tensor(0.0, device=device)

    if smooth:
        precision_scores = torch.div(
            torch.add(numerator, torch.ones(n_gram, device=device)),
            torch.add(denominator, torch.ones(n_gram, device=device)),
        )
        precision_scores[0] = numerator[0] / denominator[0]
    else:
        precision_scores = numerator / denominator

    log_precision_scores = tensor(weights, device=device) * torch.log(precision_scores)
    geometric_mean = torch.exp(torch.sum(log_precision_scores))
    brevity_penalty = tensor(1.0, device=device) if preds_len > target_len else torch.exp(1 - (target_len / preds_len))
    return brevity_penalty * geometric_mean


def bleu_score(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_gram: int = 4,
    smooth: bool = False,
    weights: Optional[Sequence[float]] = None,
) -> Tensor:
    """Calculate `BLEU score`_ of machine translated text with one or more references.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing - see [2]
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Return:
        Tensor with BLEU Score

    Raises:
        ValueError: If ``preds`` and ``target`` corpus have different lengths.
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.functional.text import bleu_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> bleu_score(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """
    preds_ = [preds] if isinstance(preds, str) else preds
    target_ = [[tgt] if isinstance(tgt, str) else tgt for tgt in target]

    if len(preds_) != len(target_):
        raise ValueError(f"Corpus has different size {len(preds_)} != {len(target_)}")

    if weights is not None and len(weights) != n_gram:
        raise ValueError(f"List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}")
    if weights is None:
        weights = [1.0 / n_gram] * n_gram

    numerator = torch.zeros(n_gram)
    denominator = torch.zeros(n_gram)
    preds_len = tensor(0.0)
    target_len = tensor(0.0)

    preds_len, target_len = _bleu_score_update(
        preds_, target_, numerator, denominator, preds_len, target_len, n_gram, _tokenize_fn
    )

    return _bleu_score_compute(preds_len, target_len, numerator, denominator, n_gram, weights, smooth)
