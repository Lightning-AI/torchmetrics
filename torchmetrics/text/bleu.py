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
from typing import Any, Callable, Optional, Sequence
from warnings import warn

import torch
from deprecate import deprecated
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update, _tokenize_fn


class BLEUScore(Metric):
    """Calculate `BLEU score`_ of machine translated text with one or more references.

    Args:
        n_gram:
            Gram value ranged from 1 to 4 (Default 4)
        smooth:
            Whether or not to apply smoothing – see [2]
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Example:
        >>> from torchmetrics import BLEUScore
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> metric = BLEUScore()
        >>> metric(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """

    is_differentiable = False
    higher_is_better = True
    preds_len: Tensor
    target_len: Tensor
    numerator: Tensor
    denominator: Tensor

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
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
        warn(
            "Input order of targets and preds were changed to predictions firsts and targets second in v0.7."
            " Warning will be removed in v0.8."
        )
        self.n_gram = n_gram
        self.smooth = smooth

        self.add_state("preds_len", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_len", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

    @deprecated(
        args_mapping={"translate_corpus": "preds", "reference_corpus": "target"},
        target=True,
        deprecated_in="0.7",
        remove_in="0.8",
    )
    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:  # type: ignore
        """Compute Precision Scores.

        Args:
            preds: An iterable of machine translated corpus
            target: An iterable of iterables of reference corpus

        .. deprecated:: v0.7
            Args:
                translate_corpus:
                    This argument is deprecated in favor of  `preds` and will be removed in v0.8.
                reference_corpus:
                    This argument is deprecated in favor of  `target` and will be removed in v0.8.
        """
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
        """Calculate BLEU score.

        Return:
            Tensor with BLEU Score
        """
        return _bleu_score_compute(
            self.preds_len, self.target_len, self.numerator, self.denominator, self.n_gram, self.smooth
        )
