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
import torch
from torch import tensor
from torch.functional import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update


class BLEUScore(Metric):
    """
    Calculate BLEU score of machine translated text with one or more references.

    Args:
        n_gram:
            Gram value ranged from 1 to 4 (Default 4)
        smooth:
            Whether or not to apply smoothing – Lin et al. 2004

    Example:
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu http://www.aclweb.org/anthology/P02-1040.pdf
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False):
        super().__init__()
        self.n_gram = n_gram
        self.smooth = smooth

        self.add_state("c", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("r", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

    def update(self, reference_corpus, translate_corpus) -> None:
        """
        Compute Precision Scores.
        Args:
            reference_corpus: An iterable of iterables of reference corpus
            translate_corpus: An iterable of machine translated corpus
        """
        self.c, self.r = _bleu_score_update(
            reference_corpus, translate_corpus, self.numerator, self.denominator, self.c, self.r, self.n_gram
        )

    def compute(self) -> Tensor:
        """
        Calculate BLEU score

        Return:
            Tensor with BLEU Score
        """
        trans_len = self.c.clone().detach()
        ref_len = self.r.clone().detach()

        return _bleu_score_compute(
            trans_len, ref_len, self.numerator, self.denominator, self.c, self.r, self.n_gram, self.smooth
        )
