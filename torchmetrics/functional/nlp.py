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

from torch import Tensor

from torchmetrics.functional.text.bleu import bleu_score as _bleu_score


def bleu_score(
    reference_corpus: Sequence[Sequence[Sequence[str]]],
    translate_corpus: Sequence[Sequence[str]],
    n_gram: int = 4,
    smooth: bool = False,
) -> Tensor:
    """Calculate `BLEU score <https://en.wikipedia.org/wiki/BLEU>`_ of machine translated text with one or more
    references.

    Example:
        >>> from torchmetrics.functional import bleu_score
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> bleu_score(reference_corpus, translate_corpus)
        tensor(0.7598)

    .. deprecated:: v0.5
        Use :func:`torchmetrics.functional.text.bleu.bleu_score`. Will be removed in v0.6.
    """
    warn(
        "Function `functional.nlp.bleu_score` is deprecated in v0.5 and will be removed in v0.6."
        " Use `functional.text.bleu.bleu_score` instead.",
        DeprecationWarning,
    )
    return _bleu_score(reference_corpus, translate_corpus, n_gram, smooth)
