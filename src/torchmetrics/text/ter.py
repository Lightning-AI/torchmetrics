# Copyright The Lightning team.
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
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TranslationEditRate.plot"]


class TranslationEditRate(Metric):
    """Calculate Translation edit rate (`TER`_)  of machine translated text with one or more references.

    This implementation follows the one from `SacreBleu_ter`_, which is a
    near-exact reimplementation of the Tercom algorithm, produces identical results on all "sane" outputs.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of hypothesis corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``ter`` (:class:`~torch.Tensor`): if ``return_sentence_level_score=True`` return a corpus-level translation
      edit rate with a list of sentence-level translation_edit_rate, else return a corpus-level translation edit rate

    Args:
        normalize: An indication whether a general tokenization to be applied.
        no_punctuation: An indication whteher a punctuation to be removed from the sentences.
        lowercase: An indication whether to enable case-insesitivity.
        asian_support: An indication whether asian characters to be processed.
        return_sentence_level_score: An indication whether a sentence-level TER to be returned.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.text import TranslationEditRate
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> ter = TranslationEditRate()
        >>> ter(preds, target)
        tensor(0.1538)
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

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
    ) -> None:
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

    def update(self, preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]]) -> None:
        """Update state with predictions and targets."""
        self.total_num_edits, self.total_tgt_len, self.sentence_ter = _ter_update(
            preds,
            target,
            self.tokenizer,
            self.total_num_edits,
            self.total_tgt_len,
            self.sentence_ter,
        )

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate the translate error rate (TER)."""
        ter = _ter_compute(self.total_num_edits, self.total_tgt_len)
        if self.sentence_ter is not None:
            return ter, torch.cat(self.sentence_ter)
        return ter

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.text import TranslationEditRate
            >>> metric = TranslationEditRate()
            >>> preds = ['the cat is on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import TranslationEditRate
            >>> metric = TranslationEditRate()
            >>> preds = ['the cat is on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
