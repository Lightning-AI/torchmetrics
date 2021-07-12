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
from typing import Sequence
from warnings import warn

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update


def bleu_score(
    reference_corpus: Sequence[Sequence[Sequence[str]]],
    translate_corpus: Sequence[Sequence[str]],
    n_gram: int = 4,
    smooth: bool = False
) -> Tensor:
    """
    Calculate `BLEU score <https://en.wikipedia.org/wiki/BLEU>`_ of machine translated text with one or more references

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
        >>> bleu_score(translate_corpus, reference_corpus)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu http://www.aclweb.org/anthology/P02-1040.pdf

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och https://aclanthology.org/P04-1077.pdf
    """
    warn(
        "Function `functional.nlp.bleu_score` will be deprecated in v0.5 and will be removed in v0.6."
        "Use `functional.text.bleu.bleu_score` instead.", DeprecationWarning
    )

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
