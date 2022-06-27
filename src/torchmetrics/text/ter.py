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

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.ter import _ter_compute, _ter_update, _TercomTokenizer
from torchmetrics.metric import Metric


class TranslationEditRate(Metric):
    """Calculate Translation edit rate (`TER`_)  of machine translated text with one or more references.

    This implementation follows the implmenetaions from
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/ter.py. The `sacrebleu` implmenetation is a
    near-exact reimplementation of the Tercom algorithm, produces identical results on all "sane" outputs.

    Args:
        normalize: An indication whether a general tokenization to be applied.
        no_punctuation: An indication whteher a punctuation to be removed from the sentences.
        lowercase: An indication whether to enable case-insesitivity.
        asian_support: An indication whether asian characters to be processed.
        return_sentence_level_score: An indication whether a sentence-level TER to be returned.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> metric = TranslationEditRate()
        >>> metric(preds, target)
        tensor(0.1538)

    References:
        [1] A Study of Translation Edit Rate with Targeted Human Annotation
        by Mathew Snover, Bonnie Dorr, Richard Schwartz, Linnea Micciulla and John Makhoul `TER`_
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    total_num_edits: Tensor
    total_tgt_len: Tensor
    sentence_ter: Optional[List[Tensor]] = None

    def __init__(
        self,
        normalize: bool = False,
        no_punctuation: bool = False,
        lowercase: bool = True,
        asian_support: bool = False,
        return_sentence_level_score: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not isinstance(normalize, bool):
            raise ValueError(f"Expected argument `normalize` to be of type boolean but got {normalize}.")
        if not isinstance(no_punctuation, bool):
            raise ValueError(f"Expected argument `no_punctuation` to be of type boolean but got {no_punctuation}.")
        if not isinstance(lowercase, bool):
            raise ValueError(f"Expected argument `lowercase` to be of type boolean but got {lowercase}.")
        if not isinstance(asian_support, bool):
            raise ValueError(f"Expected argument `asian_support` to be of type boolean but got {asian_support}.")

        self.tokenizer = _TercomTokenizer(normalize, no_punctuation, lowercase, asian_support)
        self.return_sentence_level_score = return_sentence_level_score

        self.add_state("total_num_edits", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tgt_len", tensor(0.0), dist_reduce_fx="sum")
        if self.return_sentence_level_score:
            self.add_state("sentence_ter", [], dist_reduce_fx="cat")

    def update(  # type: ignore
        self, preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]]
    ) -> None:
        """Update TER statistics.

        Args:
            preds: An iterable of hypothesis corpus.
            target: An iterable of iterables of reference corpus.
        """
        self.total_num_edits, self.total_tgt_len, self.sentence_ter = _ter_update(
            preds,
            target,
            self.tokenizer,
            self.total_num_edits,
            self.total_tgt_len,
            self.sentence_ter,
        )

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate the translate error rate (TER).

        Return:
            A corpus-level translation edit rate (TER).
            (Optionally) A list of sentence-level translation_edit_rate (TER) if ``return_sentence_level_score=True``.
        """
        ter = _ter_compute(self.total_num_edits, self.total_tgt_len)
        if self.sentence_ter is not None:
            return ter, torch.cat(self.sentence_ter)
        return ter
