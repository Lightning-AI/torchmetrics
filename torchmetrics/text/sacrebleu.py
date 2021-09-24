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
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.bleu import _bleu_score_update
from torchmetrics.functional.text.sacrebleu import _SacreBLEUTokenizer

from .bleu import BLEUScore


class SacreBLEUScore(BLEUScore):
    """Calculate `BLEU score`_ of machine translated text with one or more references. This implementation follows
    the behaviour of SacreBLEU [1] implementation from https://github.com/mjpost/sacrebleu.

    The SacreBLEU implementation differs from the NLTK BLEU implementation in tokenization techniques.

    Args:
        n_gram:
            Gram value ranged from 1 to 4 (Default 4)
        smooth:
            Whether or not to apply smoothing – see [2]
        tokenize:
            Tokenization technique to be used. (Default '13a')
            Supported tokenization: ['none', '13a', 'zh', 'intl', 'char']
        lowercase:
            If ``True``, BLEU score over lowercased text is calculated.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Example:
        >>> translate_corpus = ['the cat is on the mat']
        >>> reference_corpus = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> metric = BLEUScore()
        >>> metric(reference_corpus, translate_corpus)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """

    trans_len: Tensor
    ref_len: Tensor
    numerator: Tensor
    denominator: Tensor

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
        lowercase: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.n_gram = n_gram
        self.smooth = smooth
        self.tokenizer = _SacreBLEUTokenizer(tokenize, lowercase)

        self.add_state("trans_len", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("ref_len", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

    def update(  # type: ignore
        self, reference_corpus: Sequence[Sequence[str]], translate_corpus: Sequence[str]
    ) -> None:
        """Compute Precision Scores.

        Args:
            reference_corpus: An iterable of iterables of reference corpus
            translate_corpus: An iterable of machine translated corpus
        """
        reference_corpus = [
            [self.tokenizer(line, self.tokenize) for line in reference] for reference in reference_corpus
        ]
        translate_corpus = [self.tokenizer(line, self.tokenize) for line in translate_corpus]

        self.trans_len, self.ref_len = _bleu_score_update(
            reference_corpus,
            translate_corpus,
            self.numerator,
            self.denominator,
            self.trans_len,
            self.ref_len,
            self.n_gram,
        )
