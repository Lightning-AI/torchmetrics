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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors
# Date: 2021-11-25
# Link:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Copyright 2017 Maja Popovic

# The program is distributed under the terms
# of the GNU General Public Licence (GPL)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.helper import _validate_inputs

_EPS_SMOOTHING = tensor(1e-16)
# Taken from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py
_PUNCTUATIONS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _prepare_n_grams_dicts(
    n_char_order: int, n_word_order: int
) -> Tuple[
    Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]
]:
    """Prepare dictionaries with default zero values for total ref, hypothesis and matching chraracter and word n-grams.

    Args:
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.

    Return:
        Dictionaries with default zero values for total reference, hypothesis and matching character and word
        n-grams.
    """
    total_preds_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_preds_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_target_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_target_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_matching_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_matching_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}

    return (
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    )


def _get_characters(sentence: str, whitespace: bool) -> List[str]:
    """Split sentence into individual characters.

    Args:
        sentence: An input sentence to split.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.

    Return:
        A list of separated characters.
    """
    if whitespace:
        return list(sentence)
    return list(sentence.strip().replace(" ", ""))


def _separate_word_and_punctiation(word: str) -> List[str]:
    """Separates out punctuations from beginning and end of words for chrF.

    Adapted from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        word: An input word to be separated from a punctuation if present.

    Return:
        A list of a single word or a separated word and punctuation.
    """
    if len(word) == 1:
        return [word]

    if word[-1] in _PUNCTUATIONS:
        return [word[:-1], word[-1]]
    if word[0] in _PUNCTUATIONS:
        return [word[0], word[1:]]
    return [word]


def _get_words_and_punctiation(sentence: str) -> List[str]:
    """Separates out punctuations from beginning and end of words for chrF for all words in the sentence.

    Args:
        sentence: An input sentence to split

    Return:
        An aggregated list of separated words and punctuations.
    """
    return sum((_separate_word_and_punctiation(word) for word in sentence.strip().split()), [])


def _ngram_counts(char_or_word_list: List[str], n_gram_order: int) -> Dict[int, Dict[Tuple[str, ...], Tensor]]:
    """Calculate n-gram counts.

    Args:
        char_or_word_list: A list of characters of words
        n_gram_order: The largest number of n-gram.

    Return:
        A dictionary of dictionaries with a counts of given n-grams.
    """
    ngrams: Dict[int, Dict[Tuple[str, ...], Tensor]] = defaultdict(lambda: defaultdict(lambda: tensor(0.0)))
    for n in range(1, n_gram_order + 1):
        for ngram in (tuple(char_or_word_list[i : i + n]) for i in range(len(char_or_word_list) - n + 1)):
            ngrams[n][ngram] += tensor(1)
    return ngrams


def _get_n_grams_counts_and_total_ngrams(
    sentence: str, n_char_order: int, n_word_order: int, lowercase: bool, whitespace: bool
) -> Tuple[
    Dict[int, Dict[Tuple[str, ...], Tensor]],
    Dict[int, Dict[Tuple[str, ...], Tensor]],
    Dict[int, Tensor],
    Dict[int, Tensor],
]:
    """Get n-grams and total n-grams.

    Args:
        sentence: An input sentence
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.

    Return:
        char_n_grams_counts: A dictionary of dictionaries with sentence character n-grams.
        word_n_grams_counts: A dictionary of dictionaries with sentence word n-grams.
        total_char_n_grams: A dictionary containing a total number of sentence character n-grams.
        total_word_n_grams: A dictionary containing a total number of sentence word n-grams.
    """

    def _char_and_word_ngrams_counts(
        sentence: str, n_char_order: int, n_word_order: int, lowercase: bool
    ) -> Tuple[Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Dict[Tuple[str, ...], Tensor]]]:
        """Get a dictionary of dictionaries with a counts of given n-grams."""
        if lowercase:
            sentence = sentence.lower()
        char_n_grams_counts = _ngram_counts(_get_characters(sentence, whitespace), n_char_order)
        word_n_grams_counts = _ngram_counts(_get_words_and_punctiation(sentence), n_word_order)
        return char_n_grams_counts, word_n_grams_counts

    def _get_total_ngrams(n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]]) -> Dict[int, Tensor]:
        """Get total sum of n-grams over n-grams w.r.t n."""
        total_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
        for n in n_grams_counts:
            total_n_grams[n] = tensor(sum(n_grams_counts[n].values()))
        return total_n_grams

    char_n_grams_counts, word_n_grams_counts = _char_and_word_ngrams_counts(
        sentence, n_char_order, n_word_order, lowercase
    )
    total_char_n_grams = _get_total_ngrams(char_n_grams_counts)
    total_word_n_grams = _get_total_ngrams(word_n_grams_counts)

    return char_n_grams_counts, word_n_grams_counts, total_char_n_grams, total_word_n_grams


def _get_ngram_matches(
    hyp_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    ref_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
) -> Dict[int, Tensor]:
    """Get a number of n-gram matches between reference and hypothesis n-grams.

    Args:
        hyp_n_grams_counts: n-grams counts for hypothesis
        ref_n_grams_counts: n-grams counts for reference

    Return:
        matching_n_grams
    """
    matching_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    for n in hyp_n_grams_counts:
        matching_n_grams[n] = tensor(
            sum(
                torch.min(ref_n_grams_counts[n][n_gram], hyp_n_grams_counts[n][n_gram])
                for n_gram in hyp_n_grams_counts[n]
            )
        )
    return matching_n_grams


def _sum_over_dicts(total_n_grams: Dict[int, Tensor], n_grams: Dict[int, Tensor]) -> Dict[int, Tensor]:
    """Aggregate total n-grams to keep corpus-level statistics.

    Args:
        total_n_grams: A dictionary containing a total corpus-level number of n-grams.
        n_grams: A dictionary containing a sentence-level number of n-grams.

    Return:
        A dictionary containing a total corpus-level number of n-grams.
    """
    for n in n_grams:
        total_n_grams[n] += n_grams[n]
    return total_n_grams


def _calculate_fscore(
    matching_char_n_grams: Dict[int, Tensor],
    matching_word_n_grams: Dict[int, Tensor],
    hyp_char_n_grams: Dict[int, Tensor],
    hyp_word_n_grams: Dict[int, Tensor],
    ref_char_n_grams: Dict[int, Tensor],
    ref_word_n_grams: Dict[int, Tensor],
    n_order: float,
    beta: float,
) -> Tensor:
    """Calculate sentence-level chrF/chrF++ score.

    For given hypothesis and reference statistics (either sentence-level or corpus-level)
    the chrF/chrF++ score is returned.

    Args:
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        hyp_char_n_grams: A total number of hypothesis character n-grams.
        hyp_word_n_grams: A total number of hypothesis word n-grams.
        ref_char_n_grams: A total number of reference character n-grams.
        ref_word_n_grams: A total number of reference word n-grams.
        n_order: A sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.

    Return:
        A chrF/chrF++ score. This function is universal both for sentence-level and corpus-level calucation.
    """

    def _get_n_gram_fscore(
        matching_n_grams: Dict[int, Tensor], ref_n_grams: Dict[int, Tensor], hyp_n_grams: Dict[int, Tensor], beta: float
    ) -> Dict[int, Tensor]:
        """Get n-gram level f-score."""
        precision: Dict[int, Tensor] = {
            n: matching_n_grams[n] / hyp_n_grams[n] if hyp_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams
        }
        recall: Dict[int, Tensor] = {
            n: matching_n_grams[n] / ref_n_grams[n] if ref_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams
        }
        denominator: Dict[int, Tensor] = {
            n: torch.max(beta**2 * precision[n] + recall[n], _EPS_SMOOTHING) for n in matching_n_grams
        }
        f_score: Dict[int, Tensor] = {
            n: (1 + beta**2) * precision[n] * recall[n] / denominator[n] for n in matching_n_grams
        }

        return f_score

    char_n_gram_f_score = _get_n_gram_fscore(matching_char_n_grams, ref_char_n_grams, hyp_char_n_grams, beta)
    word_n_gram_f_score = _get_n_gram_fscore(matching_word_n_grams, ref_word_n_grams, hyp_word_n_grams, beta)

    return (sum(char_n_gram_f_score.values()) + sum(word_n_gram_f_score.values())) / tensor(n_order)


def _calculate_sentence_level_chrf_score(
    targets: List[str],
    pred_char_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    pred_word_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    pred_char_n_grams: Dict[int, Tensor],
    pred_word_n_grams: Dict[int, Tensor],
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
    lowercase: bool,
    whitespace: bool,
) -> Tuple[Tensor, Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]]:
    """Calculate the best sentence-level chrF/chrF++ score.

    For a given pre-processed hypothesis, all references are evaluated and score and statistics
    for the best matching reference is returned.

    Args:
        targets: An iterable of references.
        pred_char_n_grams_counts: A dictionary of dictionaries with hypothesis character n-grams.
        pred_word_n_grams_counts: A dictionary of dictionaries with hypothesis word n-grams.
        pred_char_n_grams: A total number of hypothesis character n-grams.
        pred_word_n_grams: A total number of hypothesis word n-grams.
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        n_order: A sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.

    Return:
        Return chrF/chrF++ score and statistics for the best matching hypothesis and reference.

        f_score: A sentence-level chrF/chrF++ score.
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        target_char_n_grams: A total number of reference character n-grams.
        target_word_n_grams: A total number of reference word n-grams.
    """
    best_f_score = tensor(0.0)
    best_matching_char_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_matching_word_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_target_char_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_target_word_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))

    for target in targets:
        (
            target_char_n_grams_counts,
            target_word_n_grams_counts,
            target_char_n_grams,
            target_word_n_grams,
        ) = _get_n_grams_counts_and_total_ngrams(target, n_char_order, n_word_order, lowercase, whitespace)
        matching_char_n_grams = _get_ngram_matches(target_char_n_grams_counts, pred_char_n_grams_counts)
        matching_word_n_grams = _get_ngram_matches(target_word_n_grams_counts, pred_word_n_grams_counts)

        f_score = _calculate_fscore(
            matching_char_n_grams,
            matching_word_n_grams,
            pred_char_n_grams,
            pred_word_n_grams,
            target_char_n_grams,
            target_word_n_grams,
            n_order,
            beta,
        )

        if f_score > best_f_score:
            best_f_score = f_score
            best_matching_char_n_grams = matching_char_n_grams
            best_matching_word_n_grams = matching_word_n_grams
            best_target_char_n_grams = target_char_n_grams
            best_target_word_n_grams = target_word_n_grams

    return (
        best_f_score,
        best_matching_char_n_grams,
        best_matching_word_n_grams,
        best_target_char_n_grams,
        best_target_word_n_grams,
    )


def _chrf_score_update(
    preds: Union[str, Sequence[str]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
    total_preds_char_n_grams: Dict[int, Tensor],
    total_preds_word_n_grams: Dict[int, Tensor],
    total_target_char_n_grams: Dict[int, Tensor],
    total_target_word_n_grams: Dict[int, Tensor],
    total_matching_char_n_grams: Dict[int, Tensor],
    total_matching_word_n_grams: Dict[int, Tensor],
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
    lowercase: bool,
    whitespace: bool,
    sentence_chrf_score: Optional[List[Tensor]] = None,
) -> Tuple[
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Optional[List[Tensor]],
]:
    """Update function for chrf score.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        total_preds_char_n_grams: A dictionary containing a total number of hypothesis character n-grams.
        total_preds_word_n_grams: A dictionary containing a total number of hypothesis word n-grams.
        total_target_char_n_grams: A dictionary containing a total number of reference character n-grams.
        total_target_word_n_grams: A dictionary containing a total number of reference word n-grams.
        total_matching_char_n_grams:
            A dictionary containing a total number of matching character n-grams between references and hypotheses.
        total_matching_word_n_grams:
            A dictionary containing a total number of total matching word n-grams between references and hypotheses.
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        n_order: Sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.
        sentence_chrf_score: A list of sentence-level chrF/chrF++ scores.

    Return:
        total_target_char_n_grams: number of reference character n-grams.
        total_target_word_n_grams: number of reference word n-grams.
        total_preds_char_n_grams: number of hypothesis character n-grams.
        total_preds_word_n_grams: number of hypothesis word n-grams.
        total_matching_char_n_grams: number of matching character n-grams between references and hypotheses.
        total_matching_word_n_grams: number of total matching word n-grams between references and hypotheses.
        sentence_chrf_score: A list of sentence-level chrF/chrF++ scores.

    Raises:
        ValueError:
            If length of ``preds`` and ``target`` differs.
    """
    target_corpus, preds = _validate_inputs(target, preds)

    for pred, targets in zip(preds, target_corpus):
        (
            pred_char_n_grams_counts,
            pred_word_n_grams_counts,
            pred_char_n_grams,
            pred_word_n_grams,
        ) = _get_n_grams_counts_and_total_ngrams(pred, n_char_order, n_word_order, lowercase, whitespace)
        total_preds_char_n_grams = _sum_over_dicts(total_preds_char_n_grams, pred_char_n_grams)
        total_preds_word_n_grams = _sum_over_dicts(total_preds_word_n_grams, pred_word_n_grams)

        (
            sentence_level_f_score,
            matching_char_n_grams,
            matching_word_n_grams,
            target_char_n_grams,
            target_word_n_grams,
        ) = _calculate_sentence_level_chrf_score(
            targets,  # type: ignore
            pred_char_n_grams_counts,
            pred_word_n_grams_counts,
            pred_char_n_grams,
            pred_word_n_grams,
            n_char_order,
            n_word_order,
            n_order,
            beta,
            lowercase,
            whitespace,
        )

        if sentence_chrf_score is not None:
            sentence_chrf_score.append(sentence_level_f_score.unsqueeze(0))

        total_target_char_n_grams = _sum_over_dicts(total_target_char_n_grams, target_char_n_grams)
        total_target_word_n_grams = _sum_over_dicts(total_target_word_n_grams, target_word_n_grams)
        total_matching_char_n_grams = _sum_over_dicts(total_matching_char_n_grams, matching_char_n_grams)
        total_matching_word_n_grams = _sum_over_dicts(total_matching_word_n_grams, matching_word_n_grams)

    return (
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        sentence_chrf_score,
    )


def _chrf_score_compute(
    total_preds_char_n_grams: Dict[int, Tensor],
    total_preds_word_n_grams: Dict[int, Tensor],
    total_target_char_n_grams: Dict[int, Tensor],
    total_target_word_n_grams: Dict[int, Tensor],
    total_matching_char_n_grams: Dict[int, Tensor],
    total_matching_word_n_grams: Dict[int, Tensor],
    n_order: float,
    beta: float,
) -> Tensor:
    """Compute chrF/chrF++ score based on pre-computed target, prediction and matching character and word n-grams.

    Args:
        total_preds_char_n_grams: number of hypothesis character n-grams.
        total_preds_word_n_grams: number of hypothesis word n-grams.
        total_target_char_n_grams: number of reference character n-grams.
        total_target_word_n_grams: number of reference word n-grams.
        total_matching_char_n_grams: number of matching character n-grams between references and hypotheses.
        total_matching_word_n_grams: number of total matching word n-grams between references and hypotheses.
        n_order: A sum of character and word n-gram order.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.

    Return:
        A corpus-level chrF/chrF++ score.
    """
    return _calculate_fscore(
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        n_order,
        beta,
    )


def chrf_score(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_char_order: int = 6,
    n_word_order: int = 2,
    beta: float = 2.0,
    lowercase: bool = False,
    whitespace: bool = False,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Calculate `chrF score`_  of machine translated text with one or more references.

    This implementation supports both chrF score computation introduced in [1] and chrF++ score introduced in
    `chrF++ score`_. This implementation follows the implmenetaions from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        n_char_order:
            A character n-gram order. If `n_char_order=6`, the metrics refers to the official chrF/chrF++.
        n_word_order:
            A word n-gram order. If `n_word_order=2`, the metric refers to the official chrF++. If `n_word_order=0`, the
            metric is equivalent to the original chrF.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insesitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.
        return_sentence_level_score: An indication whether a sentence-level chrF/chrF++ score to be returned.

    Return:
        A corpus-level chrF/chrF++ score.
        (Optionally) A list of sentence-level chrF/chrF++ scores if `return_sentence_level_score=True`.

    Raises:
        ValueError:
            If ``n_char_order`` is not an integer greater than or equal to 1.
        ValueError:
            If ``n_word_order`` is not an integer greater than or equal to 0.
        ValueError:
            If ``beta`` is smaller than 0.

    Example:
        >>> from torchmetrics.functional.text import chrf_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> chrf_score(preds, target)
        tensor(0.8640)

    References:
        [1] chrF: character n-gram F-score for automatic MT evaluation by Maja Popović `chrF score`_

        [2] chrF++: words helping character n-grams by Maja Popović `chrF++ score`_
    """
    if not isinstance(n_char_order, int) or n_char_order < 1:
        raise ValueError("Expected argument `n_char_order` to be an integer greater than or equal to 1.")
    if not isinstance(n_word_order, int) or n_word_order < 0:
        raise ValueError("Expected argument `n_word_order` to be an integer greater than or equal to 0.")
    if beta < 0:
        raise ValueError("Expected argument `beta` to be greater than 0.")

    n_order = float(n_char_order + n_word_order)

    (
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    ) = _prepare_n_grams_dicts(n_char_order, n_word_order)

    sentence_chrf_score: Optional[List[Tensor]] = [] if return_sentence_level_score else None

    (
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        sentence_chrf_score,
    ) = _chrf_score_update(
        preds,
        target,
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        n_char_order,
        n_word_order,
        n_order,
        beta,
        lowercase,
        whitespace,
        sentence_chrf_score,
    )

    chrf_f_score = _chrf_score_compute(
        total_preds_char_n_grams,
        total_preds_word_n_grams,
        total_target_char_n_grams,
        total_target_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        n_order,
        beta,
    )

    if sentence_chrf_score:
        return chrf_f_score, torch.cat(sentence_chrf_score)
    return chrf_f_score
