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
from typing import Any, Optional, Sequence

from typing_extensions import Literal

from torchmetrics.functional.text.bleu import _bleu_score_update
from torchmetrics.functional.text.sacre_bleu import _SacreBLEUTokenizer
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.utilities.imports import _REGEX_AVAILABLE

AVAILABLE_TOKENIZERS = ("none", "13a", "zh", "intl", "char")


class SacreBLEUScore(BLEUScore):
    """Calculate `BLEU score`_ of machine translated text with one or more references. This implementation follows
    the behaviour of `SacreBLEU`_.

    The SacreBLEU implementation differs from the NLTK BLEU implementation in tokenization techniques.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of machine translated corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``sacre_bleu`` (:class:`~torch.Tensor`): A tensor with the SacreBLEU Score

    Args:
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing, see `SacreBLEU`_
        tokenize: Tokenization technique to be used.
            Supported tokenization: ``['none', '13a', 'zh', 'intl', 'char']``
        lowercase:  If ``True``, BLEU score over lowercased text is calculated.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Raises:
        ValueError:
            If ``tokenize`` not one of 'none', '13a', 'zh', 'intl' or 'char'
        ValueError:
            If ``tokenize`` is set to 'intl' and `regex` is not installed
        ValueError:
            If a length of a list of weights is not ``None`` and not equal to ``n_gram``.


    Example:
        >>> from torchmetrics import SacreBLEUScore
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> sacre_bleu = SacreBLEUScore()
        >>> sacre_bleu(preds, target)
        tensor(0.7598)

    Additional References:

        - Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
          and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
        lowercase: bool = False,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ):
        super().__init__(n_gram=n_gram, smooth=smooth, weights=weights, **kwargs)
        if tokenize not in AVAILABLE_TOKENIZERS:
            raise ValueError(f"Argument `tokenize` expected to be one of {AVAILABLE_TOKENIZERS} but got {tokenize}.")

        if tokenize == "intl" and not _REGEX_AVAILABLE:
            raise ModuleNotFoundError(
                "`'intl'` tokenization requires that `regex` is installed."
                " Use `pip install regex` or `pip install torchmetrics[text]`."
            )
        self.tokenizer = _SacreBLEUTokenizer(tokenize, lowercase)

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        self.preds_len, self.target_len = _bleu_score_update(
            preds,
            target,
            self.numerator,
            self.denominator,
            self.preds_len,
            self.target_len,
            self.n_gram,
            self.tokenizer,
        )
