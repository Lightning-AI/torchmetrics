# Copyright 2017 Maja Popovic

# The program is distributed under the terms
# of the GNU General Public Licence (GPL)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Publications of results obtained through the use of original or
# modified versions of the software have to cite the authors by refering
# to the following publication:

# Maja Popović (2015).
# "chrF: character n-gram F-score for automatic MT evaluation".
# In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT15), pages 392–395
# Lisbon, Portugal, September 2015.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor

_EPS_SMOOTHING = tensor(1e-16)
# Taken from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py
_PUNCTUATIONS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _prepare_n_grams_dicts(
    n_char_order: int, n_word_order: int
) -> Tuple[
    Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]
]:
    """Prepare dictionaries dictionaries with default zero values for total reference, hypothesis and matching
    character and word n-grams.

    Args:
        n_char_order:
            A character n-gram order.
        n_word_order:
            A word n-gram order.

    Return:
        Dictionaries with default zero values for total reference, hypothesis and matching character and word
        n-grams.
    """
    total_ref_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_ref_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_hyp_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_hyp_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_matching_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_matching_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}

    return (
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    )


def _defaultdict_of_tensors_tuple_keys() -> Dict[Tuple[str, ...], Tensor]:
    """A wrapper for creating `defaultdict` with key type of `Tuple[str, ...]` and initialized with a zero
    tensor."""

    def zero_tensor() -> Tensor:
        return tensor(0.0)

    return defaultdict(zero_tensor)


def _defaultdict_of_tensors_int_keys() -> Dict[int, Tensor]:
    """A wrapper for creating `defaultdict` with key type of `int` and initialized with a zero tensor."""

    def zero_tensor() -> Tensor:
        return tensor(0.0)

    return defaultdict(zero_tensor)


def _get_characters(sentence: str) -> List[str]:
    """Split sentence into individual characters.

    Args:
        sentence:
            An input sentence to split.

    Return:
        A list of separated characters.
    """
    return list(sentence.strip().replace(" ", ""))


def _separate_word_and_punctiation(word: str) -> List[str]:
    """
    Separates out punctuations from beginning and end of words for chrF. Adapted from https://github.com/m-popovic/chrF
    and https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.
    Args:
        word:
            An input word to be separated from a punctuation if present.

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
        sentence:
            An input sentence to split

    Return:
        A list of separated words and punctuations..
    """
    return sum((_separate_word_and_punctiation(word) for word in sentence.strip().split()), [])


def _ngram_counts(char_or_word_list: List[str], n_gram_order: int) -> Dict[int, Dict[Tuple[str, ...], Tensor]]:
    """
    Args:
        char_or_word_list:
            A list of characters of words
        n_gram_order:
            A largest number of n-gram.

    Return:
        A dictionary of dictionaries with a counts of given n-grams.
    """
    ngrams: Dict[int, Dict[Tuple[str, ...], Tensor]] = defaultdict(_defaultdict_of_tensors_tuple_keys)
    for n in range(1, n_gram_order + 1):
        for ngram in (tuple(char_or_word_list[i : i + n]) for i in range(len(char_or_word_list) - n + 1)):
            ngrams[n][ngram] += tensor(1)
    return ngrams


def _get_n_grams_counts_and_total_ngrams(
    sentence: str, n_char_order: int, n_word_order: int, lowercase: bool
) -> Tuple[
    Dict[int, Dict[Tuple[str, ...], Tensor]],
    Dict[int, Dict[Tuple[str, ...], Tensor]],
    Dict[int, Tensor],
    Dict[int, Tensor],
]:
    """
    Args:
        sentence:
            An input sentence
        n_char_order:
            A character n-gram order.
        n_word_order:
            A word n-gram order.
        lowercase:
            An indication whether to enable case-insesitivity.

    Return:
        char_n_grams_counts:
        word_n_grams_counts:
        total_char_n_grams:
        total_word_n_grams:
    """

    def _char_and_word_ngrams_counts(
        sentence: str, n_char_order: int, n_word_order: int, lowercase: bool
    ) -> Tuple[Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Dict[Tuple[str, ...], Tensor]]]:
        """Get a dictionary of dictionaries with a counts of given n-grams."""
        if lowercase:
            sentence = sentence.lower()
        char_n_grams_counts = _ngram_counts(_get_characters(sentence), n_char_order)
        word_n_grams_counts = _ngram_counts(_get_words_and_punctiation(sentence), n_word_order)
        return char_n_grams_counts, word_n_grams_counts

    def _get_total_ngrams(n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]]) -> Dict[int, Tensor]:
        """Get total sum of n-grams over n-grams w.r.t n."""
        total_n_grams: Dict[int, Tensor] = _defaultdict_of_tensors_int_keys()
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
    ref_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    hyp_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
) -> Dict[int, Tensor]:
    """Get a number of n-gram matches between reference and hypothesis n-grams.

    Args:
        ref_n_grams_counts:
        ref_n_grams_counts:

    Return:
    """
    matching_n_grams = _defaultdict_of_tensors_int_keys()
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
        total_n_grams:
            A dictionary containing a total corpus-level number of n-grams.
        n_grams:
            A dictionary containing a sentence-level number of n-grams.

    Return:
        A dictionary containing a total corpus-level number of n-grams.
    """
    for n in n_grams:
        total_n_grams[n] += n_grams[n]
    return total_n_grams


def _calculate_fscore(
    matching_char_n_grams: Dict[int, Tensor],
    matching_word_n_grams: Dict[int, Tensor],
    ref_char_n_grams: Dict[int, Tensor],
    ref_word_n_grams: Dict[int, Tensor],
    hyp_char_n_grams: Dict[int, Tensor],
    hyp_word_n_grams: Dict[int, Tensor],
    n_order: float,
    beta: float,
) -> Tensor:
    """Calculate sentence-level ChrF/ChrF++ score. For given hypothesis and reference statistics (either sentence-
    level or corpus-level) the ChrF/ChrF++ score is returned.

    Args:
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        ref_char_n_grams:
            A total number of reference character n-grams.
        ref_word_n_grams:
            A total number of reference word n-grams.
        hyp_char_n_grams:
            A total number of hypothesis character n-grams.
        hyp_word_n_grams:
            A total number of hypothesis word n-grams.
        n_order:
            A sum of character and word n-gram order.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.

    Return:
        A ChrF/ChrF++ score. This function is universal both for sentence-level and corpus-level calucation.
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
            n: torch.max(beta ** 2 * precision[n] + recall[n], _EPS_SMOOTHING) for n in matching_n_grams
        }
        f_score: Dict[int, Tensor] = {
            n: (1 + beta ** 2) * precision[n] * recall[n] / denominator[n] for n in matching_n_grams
        }

        return f_score

    char_n_gram_f_score = _get_n_gram_fscore(matching_char_n_grams, ref_char_n_grams, hyp_char_n_grams, beta)
    word_n_gram_f_score = _get_n_gram_fscore(matching_word_n_grams, ref_word_n_grams, hyp_word_n_grams, beta)

    f_score = (sum(char_n_gram_f_score.values()) + sum(word_n_gram_f_score.values())) / tensor(n_order)  # type: ignore
    return f_score


def _calculate_sentence_level_chrf_score(
    references: List[str],
    hyp_char_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    hyp_word_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]],
    hyp_char_n_grams: Dict[int, Tensor],
    hyp_word_n_grams: Dict[int, Tensor],
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
    lowercase: bool,
) -> Tuple[Tensor, Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]]:
    """Calculate the best sentence-level ChrF/ChrF++ score. For a given pre-processed hypothesis, all references
    are evaluated and score and statistics for the best matching reference is returned.

    Args:
        references:
            An iterable of references.
        hyp_char_n_grams_counts:
            A dictionary of dictionaries with hypothesis character n-grams.
        hyp_word_n_grams_counts:
            A dictionary of dictionaries with hypothesis word n-grams.
        hyp_char_n_grams:
            A total number of hypothesis character n-grams.
        hyp_word_n_grams:
            A total number of hypothesis word n-grams.
        n_char_order:
            A character n-gram order.
        n_word_order:
            A word n-gram order.
        n_order:
            A sum of character and word n-gram order.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase:
            An indication whether to enable case-insesitivity.

    Return:
        Return ChrF/ChrF++ score and statistics for the best matching hypothesis and reference.

        f_score:
            A sentence-level ChrF/ChrF++ score.
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        ref_char_n_grams:
            A total number of reference character n-grams.
        ref_word_n_grams:
            A total number of reference word n-grams.
    """

    best_f_score = tensor(0.0)
    best_matching_char_n_grams: Dict[int, Tensor] = _defaultdict_of_tensors_int_keys()
    best_matching_word_n_grams: Dict[int, Tensor] = _defaultdict_of_tensors_int_keys()
    best_ref_char_n_grams: Dict[int, Tensor] = _defaultdict_of_tensors_int_keys()
    best_ref_word_n_grams: Dict[int, Tensor] = _defaultdict_of_tensors_int_keys()

    for reference in references:
        (
            ref_char_n_grams_counts,
            ref_word_n_grams_counts,
            ref_char_n_grams,
            ref_word_n_grams,
        ) = _get_n_grams_counts_and_total_ngrams(reference, n_char_order, n_word_order, lowercase)
        matching_char_n_grams = _get_ngram_matches(ref_char_n_grams_counts, hyp_char_n_grams_counts)
        matching_word_n_grams = _get_ngram_matches(ref_word_n_grams_counts, hyp_word_n_grams_counts)

        f_score = _calculate_fscore(
            matching_char_n_grams,
            matching_word_n_grams,
            ref_char_n_grams,
            ref_word_n_grams,
            hyp_char_n_grams,
            hyp_word_n_grams,
            n_order,
            beta,
        )

        if f_score > best_f_score:
            best_f_score = f_score
            best_matching_char_n_grams = matching_char_n_grams
            best_matching_word_n_grams = matching_word_n_grams
            best_ref_char_n_grams = ref_char_n_grams
            best_ref_word_n_grams = ref_word_n_grams

    return (
        best_f_score,
        best_matching_char_n_grams,
        best_matching_word_n_grams,
        best_ref_char_n_grams,
        best_ref_word_n_grams,
    )


def _chrf_score_update(
    reference_corpus: Union[Sequence[str], Sequence[Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
    total_ref_char_n_grams: Dict[int, Tensor],
    total_ref_word_n_grams: Dict[int, Tensor],
    total_hyp_char_n_grams: Dict[int, Tensor],
    total_hyp_word_n_grams: Dict[int, Tensor],
    total_matching_char_n_grams: Dict[int, Tensor],
    total_matching_word_n_grams: Dict[int, Tensor],
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
    lowercase: bool,
    sentence_chrf_score: Optional[List[Tensor]] = None,
) -> Tuple[
    Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]
]:
    """
    Args:
        reference_corpus:
            An iterable of iterables of reference corpus.
        hypothesis_corpus:
            An iterable of hypothesis corpus.
        total_ref_char_n_grams:
            A dictionary containing total reference character n-grams.
        total_ref_word_n_grams:
            A dictionary containing total reference word n-grams.
        total_hyp_char_n_grams:
            A dictionary containing total hypothesis character n-grams.
        total_hyp_word_n_grams:
            A dictionary containing total hypothesis word n-grams.
        total_matching_char_n_grams:
            A dictionary containing total matching character n-grams between references and hypotheses.
        total_matching_word_n_grams:
            A dictionary containing total matching word n-grams between references and hypotheses.
        n_char_order:
            A character n-gram order.
        n_word_order:
            A word n-gram order.
        n_order:
            Sum of charachter and word n-gram order.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase:
            An indication whether to enable case-insesitivity.
        sentence_chrf_score:
            A list of sentence-level ChrF/ChrF++ scores.

    Return:
        total_ref_char_n_grams:
            An updated dictionary containing total reference character n-grams.
        total_ref_word_n_grams:
            An updated dictionary containing total reference word n-grams.
        total_hyp_char_n_grams:
            An updated dictionary containing total hypothesis character n-grams.
        total_hyp_word_n_grams:
            An updated dictionary containing total hypothesis word n-grams.
        total_matching_char_n_grams:
            An updated dictionary containing total matching character n-grams between references and hypotheses.
        total_matching_word_n_grams:
            An updated dictionary containing total matching word n-grams between references and hypotheses.

    Raises:
        ValueError:
            If length of `reference_corpus` and `hypothesis_corpus` differs.
    """
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]

    # Ensure reference corpus is properly of a type Sequence[Sequence[str]]
    if all(isinstance(ref, str) for ref in reference_corpus):
        if len(hypothesis_corpus) == 1:
            reference_corpus = [reference_corpus]  # type: ignore
        else:
            reference_corpus = [[ref] for ref in reference_corpus]  # type: ignore

    if hypothesis_corpus and all(ref for ref in reference_corpus) and len(reference_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")

    for (references, hypothesis) in zip(reference_corpus, hypothesis_corpus):
        (
            hyp_char_n_grams_counts,
            hyp_word_n_grams_counts,
            hyp_char_n_grams,
            hyp_word_n_grams,
        ) = _get_n_grams_counts_and_total_ngrams(hypothesis, n_char_order, n_word_order, lowercase)
        total_hyp_char_n_grams = _sum_over_dicts(total_hyp_char_n_grams, hyp_char_n_grams)
        total_hyp_word_n_grams = _sum_over_dicts(total_hyp_word_n_grams, hyp_word_n_grams)

        (
            sentence_level_f_score,
            matching_char_n_grams,
            matching_word_n_grams,
            ref_char_n_grams,
            ref_word_n_grams,
        ) = _calculate_sentence_level_chrf_score(
            references,  # type: ignore
            hyp_char_n_grams_counts,
            hyp_word_n_grams_counts,
            hyp_char_n_grams,
            hyp_word_n_grams,
            n_char_order,
            n_word_order,
            n_order,
            beta,
            lowercase,
        )

        total_ref_char_n_grams = _sum_over_dicts(total_ref_char_n_grams, ref_char_n_grams)
        total_ref_word_n_grams = _sum_over_dicts(total_ref_word_n_grams, ref_word_n_grams)
        total_matching_char_n_grams = _sum_over_dicts(total_matching_char_n_grams, matching_char_n_grams)
        total_matching_word_n_grams = _sum_over_dicts(total_matching_word_n_grams, matching_word_n_grams)

    return (
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    )


def _chrf_score_compute(
    total_ref_char_n_grams: Dict[int, Tensor],
    total_ref_word_n_grams: Dict[int, Tensor],
    total_hyp_char_n_grams: Dict[int, Tensor],
    total_hyp_word_n_grams: Dict[int, Tensor],
    total_matching_char_n_grams: Dict[int, Tensor],
    total_matching_word_n_grams: Dict[int, Tensor],
    n_order: float,
    beta: float,
) -> Tensor:
    """Compute ChrF/ChrF++ score based on pre-computed reference, hypothesis and matching character and word
    n-grams.

    Args:
        total_ref_char_n_grams:
            A dictionary containing total reference character n-grams.
        total_ref_word_n_grams:
            A dictionary containing total reference word n-grams.
        total_hyp_char_n_grams:
            A dictionary containing total hypothesis character n-grams.
        total_hyp_word_n_grams:
            A dictionary containing total hypothesis word n-grams.
        total_matching_char_n_grams:
            A dictionary containing total matching character n-grams between references and hypotheses.
        total_matching_word_n_grams:
            A dictionary containing total matching word n-grams between references and hypotheses.
        n_order:
            A sum of charachter and word n-gram order.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.

    Return:
        A corpus-level ChrF/ChrF++ score.
    """
    chrf_f_score = _calculate_fscore(
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        n_order,
        beta,
    )
    return chrf_f_score


def chrf_score(
    reference_corpus: Union[Sequence[str], Sequence[Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
    n_char_order: int = 6,
    n_word_order: int = 2,
    beta: float = 2.0,
    lowercase: bool = False,
    whitespace: bool = False,
    return_sentence_level_score: bool = False,
) -> Tensor:
    """Calculate `ChrF score`_ [1] of machine translated text with one or more references. This implementation
    supports both ChrF score computation introduced in [1] and ChrF++ score introduced in [2]. This implementation
    follows the implmenetaions from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus.
        hypothesis_corpus:
            An iterable of hypothesis corpus.
        n_char_order:
            A character n-gram order. If `n_char_order=6`, the metrics refers to the official ChrF/ChrF++.
        n_word_order:
            A word n-gram order. If `n_word_order=2`, the metric refers to the official ChrF++. If `n_word_order=0`, the
            metric is equivalent to the original ChrF.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase:
            An indication whether to enable case-insesitivity.
        whitespace:
            An indication whether keep whitespaces during n-gram extraction.
        return_sentence_level_score:
            An indication whether a sentence-level ChrF/ChrF++ score to be returned.

    Return:
        A corpus-level ChrF/ChrF++ score.
        (Optionally) A list of sentence-level ChrF/ChrF++ scores.

    Example:
        >>> from torchmetrics.functional import chrf_score
        >>> hypothesis_corpus = ['the cat is on the mat']
        >>> reference_corpus = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> chrf_score(reference_corpus, hypothesis_corpus)
        tensor(0.8640)

    References:
    [1] CHRF: character n-gram F-score for automatic MT evaluation by Maja Popović
    [2] CHRF++: words helping character n-grams by Maja Popović
    """
    n_order = float(n_char_order + n_word_order)

    (
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    ) = _prepare_n_grams_dicts(n_char_order, n_word_order)

    sentence_chrf_score: Optional[List[Tensor]] = [] if return_sentence_level_score else None

    (
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
    ) = _chrf_score_update(
        reference_corpus,
        hypothesis_corpus,
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        n_char_order,
        n_word_order,
        n_order,
        beta,
        lowercase,
        sentence_chrf_score,
    )

    chrf_f_score = _chrf_score_compute(
        total_ref_char_n_grams,
        total_ref_word_n_grams,
        total_hyp_char_n_grams,
        total_hyp_word_n_grams,
        total_matching_char_n_grams,
        total_matching_word_n_grams,
        n_order,
        beta,
    )

    return chrf_f_score
