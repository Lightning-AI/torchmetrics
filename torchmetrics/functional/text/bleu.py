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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors and @sluks
# Date: 2020-07-18
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#bleu_score
from collections import Counter
from typing import Sequence, Tuple

import torch
from torch import Tensor, tensor


def _count_ngram(ngram_input_list: Sequence[str], n_gram: int) -> Counter:
    """Counting how many times each word appears in a given text with ngram.

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


def _bleu_score_update(
    reference_corpus: Sequence[Sequence[Sequence[str]]],
    translate_corpus: Sequence[Sequence[str]],
    numerator: Tensor,
    denominator: Tensor,
    trans_len: Tensor,
    ref_len: Tensor,
    n_gram: int = 4,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute the BLEU score.

    Args:
        reference_corpus: An iterable of iterables of reference corpus
        translate_corpus: An iterable of machine translated corpus
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        trans_len: count of words in a candidate translation
        ref_len: count of words in a reference translation
        n_gram: gram value ranged 1 to 4
    """

    for (translation, references) in zip(translate_corpus, reference_corpus):
        trans_len += len(translation)
        ref_len_list = [len(ref) for ref in references]
        ref_len_diff = [abs(len(translation) - x) for x in ref_len_list]
        ref_len += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
        translation_counter: Counter = _count_ngram(translation, n_gram)
        reference_counter: Counter = Counter()

        for ref in references:
            reference_counter |= _count_ngram(ref, n_gram)

        ngram_counter_clip = translation_counter & reference_counter

        for counter_clip in ngram_counter_clip:
            numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

        for counter in translation_counter:
            denominator[len(counter) - 1] += translation_counter[counter]

    return trans_len, ref_len


def _bleu_score_compute(
    trans_len: Tensor, ref_len: Tensor, numerator: Tensor, denominator: Tensor, n_gram: int = 4, smooth: bool = False
) -> Tensor:
    """Computes the BLEU score.

    Args:
        trans_len: count of words in a candidate translation
        ref_len: count of words in a reference translation
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        n_gram: gram value ranged 1 to 4
        smooth: Whether or not to apply smoothing
    """
    if min(numerator) == 0.0:
        return tensor(0.0)

    if smooth:
        precision_scores = torch.add(numerator, torch.ones(n_gram)) / torch.add(denominator, torch.ones(n_gram))
        precision_scores[0] = numerator[0] / denominator[0]
    else:
        precision_scores = numerator / denominator

    log_precision_scores = tensor([1.0 / n_gram] * n_gram) * torch.log(precision_scores)
    geometric_mean = torch.exp(torch.sum(log_precision_scores))
    brevity_penalty = tensor(1.0) if trans_len > ref_len else torch.exp(1 - (ref_len / trans_len))
    bleu = brevity_penalty * geometric_mean

    return bleu


def bleu_score(
    reference_corpus: Sequence[Sequence[Sequence[str]]],
    translate_corpus: Sequence[Sequence[str]],
    n_gram: int = 4,
    smooth: bool = False,
) -> Tensor:
    """Calculate `BLEU score <https://en.wikipedia.org/wiki/BLEU>`_ of machine translated text with one or more
    references.

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus
        translate_corpus:
            An iterable of machine translated corpus
        n_gram:
            Gram value ranged from 1 to 4 (Default 4)
        smooth:
            Whether or not to apply smoothing – see [2]

    Return:
        Tensor with BLEU Score

    Example:
        >>> from torchmetrics.functional import bleu_score
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> bleu_score(reference_corpus, translate_corpus)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu http://www.aclweb.org/anthology/P02-1040.pdf

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och https://aclanthology.org/P04-1077.pdf
    """

    if len(translate_corpus) != len(reference_corpus):
        raise ValueError(f"Corpus has different size {len(translate_corpus)} != {len(reference_corpus)}")
    numerator = torch.zeros(n_gram)
    denominator = torch.zeros(n_gram)
    trans_len = tensor(0, dtype=torch.float)
    ref_len = tensor(0, dtype=torch.float)

    trans_len, ref_len = _bleu_score_update(
        reference_corpus, translate_corpus, numerator, denominator, trans_len, ref_len, n_gram
    )

    return _bleu_score_compute(trans_len, ref_len, numerator, denominator, n_gram, smooth)
