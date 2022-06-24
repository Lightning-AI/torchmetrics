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
# Date: 2021-11-25
# Link:

import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.text.chrf import _chrf_score_compute, _chrf_score_update, _prepare_n_grams_dicts

_N_GRAM_LEVELS = ("char", "word")
_TEXT_LEVELS = ("preds", "target", "matching")

_DICT_STATES_NAMES = (
    "total_preds_char_n_grams",
    "total_preds_word_n_grams",
    "total_target_char_n_grams",
    "total_target_word_n_grams",
    "total_matching_char_n_grams",
    "total_matching_word_n_grams",
)

_DICT_STATES_TYPES = Tuple[
    Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]
]


class CHRFScore(Metric):
    """Calculate `chrf score`_ of machine translated text with one or more references.

    This implementation supports both ChrF score computation introduced in [1] and chrF++ score introduced
    in `chrF++ score_`. This implementation follows the implmenetaions from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        n_char_order: A character n-gram order. If ``n_char_order=6``, the metrics refers to the official chrF/chrF++.
        n_word_order: A word n-gram order. If ``n_word_order=2``, the metric refers to the official chrF++.
            If ``n_word_order=0``, the metric is equivalent to the original ChrF.
        beta: parameter determining an importance of recall w.r.t. precision. If ``beta=1``, their importance is equal.
        lowercase: An indication whether to enable case-insesitivity.
        whitespace: An indication whether keep whitespaces during n-gram extraction.
        return_sentence_level_score: An indication whether a sentence-level chrF/chrF++ score to be returned.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``n_char_order`` is not an integer greater than or equal to 1.
        ValueError:
            If ``n_word_order`` is not an integer greater than or equal to 0.
        ValueError:
            If ``beta`` is smaller than 0.

    Example:
        >>> from torchmetrics import CHRFScore
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> metric = CHRFScore()
        >>> metric(preds, target)
        tensor(0.8640)

    References:
        [1] chrF: character n-gram F-score for automatic MT evaluation by Maja Popović `chrF score`_

        [2] chrF++: words helping character n-grams by Maja Popović `chrF++ score`_
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    sentence_chrf_score: Optional[List[Tensor]] = None

    def __init__(
        self,
        n_char_order: int = 6,
        n_word_order: int = 2,
        beta: float = 2.0,
        lowercase: bool = False,
        whitespace: bool = False,
        return_sentence_level_score: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if not isinstance(n_char_order, int) or n_char_order < 1:
            raise ValueError("Expected argument `n_char_order` to be an integer greater than or equal to 1.")
        self.n_char_order = n_char_order
        if not isinstance(n_word_order, int) or n_word_order < 0:
            raise ValueError("Expected argument `n_word_order` to be an integer greater than or equal to 0.")
        self.n_word_order = n_word_order
        if beta < 0:
            raise ValueError("Expected argument `beta` to be greater than 0.")
        self.beta = beta
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.return_sentence_level_score = return_sentence_level_score

        self.n_order = float(n_char_order + n_word_order)

        # Adding state dynamically
        for (n_gram_level, n_gram_order), text in self._get_text_n_gram_iterator():
            for n in range(1, n_gram_order + 1):
                state_name = self._get_state_name(text, n_gram_level, n)
                self.add_state(state_name, tensor(0.0), dist_reduce_fx="sum")

        if self.return_sentence_level_score:
            self.add_state("sentence_chrf_score", [], dist_reduce_fx="cat")

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:  # type: ignore
        """Compute Precision Scores.

        Args:
            preds: An iterable of hypothesis corpus.
            target: An iterable of iterables of reference corpus.
        """
        n_grams_dicts_tuple = _chrf_score_update(
            preds,
            target,
            *self._convert_states_to_dicts(),
            self.n_char_order,
            self.n_word_order,
            self.n_order,
            self.beta,
            self.lowercase,
            self.whitespace,
            self.sentence_chrf_score if self.return_sentence_level_score else None,
        )
        self._update_states_from_dicts(n_grams_dicts_tuple[:-1])
        if self.sentence_chrf_score is not None:
            self.sentence_chrf_score = n_grams_dicts_tuple[-1]

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate chrF/chrF++ score.

        Return:
            A corpus-level chrF/chrF++ score.
            (Optionally) A list of sentence-level chrF/chrF++ scores if `return_sentence_level_score=True`.
        """
        if self.sentence_chrf_score is not None:
            return (
                _chrf_score_compute(*self._convert_states_to_dicts(), self.n_order, self.beta),
                torch.cat(self.sentence_chrf_score),
            )
        return _chrf_score_compute(*self._convert_states_to_dicts(), self.n_order, self.beta)

    def _convert_states_to_dicts(self) -> _DICT_STATES_TYPES:
        """Convert global metric states to the n-gram dictionaries to be passed in `_chrf_score_update`."""
        n_grams_dicts: Dict[str, Dict[int, Tensor]] = {
            name: n_gram_dict
            for name, n_gram_dict in zip(
                _DICT_STATES_NAMES, _prepare_n_grams_dicts(self.n_char_order, self.n_word_order)
            )
        }

        for (n_gram_level, n_gram_order), text in self._get_text_n_gram_iterator():
            for n in range(1, n_gram_order + 1):
                dict_name = self._get_dict_name(text, n_gram_level)
                state_name = self._get_state_name(text, n_gram_level, n)

                n_grams_dicts[dict_name][n] = getattr(self, state_name)

        return tuple(n_grams_dicts.values())  # type: ignore

    def _update_states_from_dicts(self, n_grams_dicts_tuple: _DICT_STATES_TYPES) -> None:
        """Update global metric states based on the n-gram dictionaries calculated on the current batch."""
        n_grams_dicts = {name: n_gram_dict for name, n_gram_dict, in zip(_DICT_STATES_NAMES, n_grams_dicts_tuple)}
        for (n_gram_level, n_gram_order), text in self._get_text_n_gram_iterator():
            for n in range(1, n_gram_order + 1):
                dict_name = self._get_dict_name(text, n_gram_level)
                state_name = self._get_state_name(text, n_gram_level, n)

                setattr(self, state_name, n_grams_dicts[dict_name][n])

    @staticmethod
    def _get_dict_name(text: str, n_gram_level: str) -> str:
        """Return a dictionary name w.r.t input args."""
        return f"total_{text}_{n_gram_level}_n_grams"

    @staticmethod
    def _get_state_name(text: str, n_gram_level: str, n: int) -> str:
        """Return a metric state name w.r.t input args."""
        return f"total_{text}_{n_gram_level}_{n}_grams"

    def _get_text_n_gram_iterator(self) -> Iterator[Tuple[Tuple[str, int], str]]:
        """Get iterator over char/word and reference/hypothesis/matching n-gram level."""
        return itertools.product(zip(_N_GRAM_LEVELS, [self.n_char_order, self.n_word_order]), _TEXT_LEVELS)
