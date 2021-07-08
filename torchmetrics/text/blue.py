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
from typing import List

import torch
from torch import tensor
from torchmetrics import Metric


def _count_ngram(ngram_input_list: List[str], n_gram: int) -> Counter:
    """
    Counting how many times each word appears in a given text with ngram
    Args:
        ngram_input_list: A list of translated text or reference texts
        n_gram: gram value ranged 1 to 4
    Return:
        ngram_counter: a collections.Counter object of ngram
    """

    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(ngram_input_list) - i + 1):
            ngram_key = tuple(ngram_input_list[j:(i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter


class BLEUScore(Metric):
    """
    Calculate BLEU score of machine translated text with one or more references.
    Example:
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        tensor(0.7598)
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False):
        """
        Args:
            n_gram: Gram value ranged from 1 to 4 (Default 4)
            smooth: Whether or not to apply smoothing – Lin et al. 2004
        """
        super().__init__()
        self.n_gram = n_gram
        self.smooth = smooth

        self.add_state("c", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("r", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

    def compute(self):

        trans_len = self.c.clone().detach()
        ref_len = self.r.clone().detach()

        if min(self.numerator) == 0.0:
            return tensor(0.0, device=self.r.device)

        if self.smooth:
            precision_scores = (self.numerator + 1.0) / (self.denominator + 1.0)
        else:
            precision_scores = self.numerator / self.denominator

        log_precision_scores = tensor([1.0 / self.n_gram] * self.n_gram,
                                      device=self.r.device) * torch.log(precision_scores)
        geometric_mean = torch.exp(torch.sum(log_precision_scores))
        brevity_penalty = (
            tensor(1.0, device=self.r.device) if self.c > self.r else torch.exp(1 - (ref_len / trans_len))
        )
        bleu = brevity_penalty * geometric_mean
        return bleu

    def update(self, translate_corpus, reference_corpus) -> None:
        """
        Actual metric computation
        Args:
            translate_corpus: An iterable of machine translated corpus
            reference_corpus: An iterable of iterables of reference corpus
        """
        for (translation, references) in zip(translate_corpus, reference_corpus):
            self.c += len(translation)
            ref_len_list = [len(ref) for ref in references]
            ref_len_diff = [abs(len(translation) - x) for x in ref_len_list]
            self.r += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
            translation_counter = _count_ngram(translation, self.n_gram)
            reference_counter = Counter()

            for ref in references:
                reference_counter |= _count_ngram(ref, self.n_gram)

            ngram_counter_clip = translation_counter & reference_counter

            for counter_clip in ngram_counter_clip:
                self.numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

            for counter in translation_counter:
                self.denominator[len(counter) - 1] += translation_counter[counter]
