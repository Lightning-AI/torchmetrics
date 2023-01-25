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

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update, _tokenize_fn


class BLEUScore(Metric):
    """Calculate `BLEU score`_ of machine translated text with one or more references.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of machine translated corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``update`` the metric returns the following output:

    - ``bleu`` (:class:`~torch.Tensor`): A tensor with the BLEU Score

    Args:
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether or not to apply smoothing, see `Machine Translation Evolution`_
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Raises:
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics import BLEUScore
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> bleu = BLEUScore()
        >>> bleu(preds, target)
        tensor(0.7598)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    preds_len: Tensor
    target_len: Tensor
    numerator: Tensor
    denominator: Tensor

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.smooth = smooth
        if weights is not None and len(weights) != n_gram:
            raise ValueError(f"List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}")
        self.weights = weights if weights is not None else [1.0 / n_gram] * n_gram

        self.add_state("preds_len", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_len", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

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
            _tokenize_fn,
        )

    def compute(self) -> Tensor:
        """Calculate BLEU score."""
        return _bleu_score_compute(
            self.preds_len, self.target_len, self.numerator, self.denominator, self.n_gram, self.weights, self.smooth
        )
