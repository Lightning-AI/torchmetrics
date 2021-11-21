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

import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import tensor, Tensor

from torchmetrics.functional.classification.precision_recall import precision


def _get_characters(sentence: str) -> List[str]:
    """
    Split sentence into individual characters.

    Args:
        sentence:
            An input sentence to split.

    Return:
        A list of separated characters
    """
    return list(sentence.strip().replace(" ", ""))


def _separare_word_and_punctiation(word: str) -> List[str]:
    """
    Args:
        word

    Return:
    """
    if len(word) == 1:
        return [word]

    first_char, last_char = word[0], word[-1]
    if last_char in string.punctuation:
        return [word[:-1], last_char]
    if first_char in string.punctuation:
        return [first_char, word[1:]]
    return [word]



def _get_words_and_punctiation(sentence: str) -> List[str]:
    """"""
    return sum([_get_words_and_punctiation(word) for word in sentence.strip().split()], [])


def _ngram_counts(char_or_word_list: List[str], n_gram_order: int) -> Dict[int, Counter]:
    """
    Args:
        char_or_word_list:
        order:

    Return:
    """
    ngrams: Dict[int, Counter] = defaultdict(Counter)
    for n in range(1, n_gram_order + 1):
        for ngram in (tuple(char_or_word_list[i : i + n]) for i in range(len(char_or_word_list) - n + 1)):
            ngrams[n][ngram] += 1
    return ngrams


def _get_n_grams_counts_and_total_ngrams(
    sentence: str, n_char_order: int, n_word_order: int
) -> Tuple[Dict[int, Counter], Dict[int, Counter], Counter, Counter]:
    """"""
    def _char_and_word_ngrams_counts(
        sentence: str, n_char_order: int, n_word_order: int
    ) -> Tuple[Dict[int, Counter], Dict[int, Counter]]:
        """"""
        char_n_grams_counts = _ngram_counts(_get_characters(sentence), n_char_order)
        word_n_grams_counts = _ngram_counts(_get_words_and_punctiation(sentence), n_word_order)
        return char_n_grams_counts, word_n_grams_counts

    def _get_total_ngrams(n_grams_counts: Dict[int, Counter]) -> Counter:
        """"""
        total_n_grams: Counter = Counter()
        for n in n_grams_counts:
            total_n_grams[n] = sum(n_grams_counts[n].values())
        return total_n_grams

    char_n_grams_counts, word_n_grams_counts = _char_and_word_ngrams_counts(sentence, n_char_order, n_word_order)
    total_char_n_grams = _get_total_ngrams(char_n_grams_counts)
    total_word_n_grams = _get_total_ngrams(word_n_grams_counts)
    
    return char_n_grams_counts, word_n_grams_counts, total_char_n_grams, total_word_n_grams


def _get_ngram_matches(ref_n_grams_counts: Dict[int, Counter], hyp_n_grams_counts: Dict[int, Counter]) -> Counter:
    """
    Args:

    Return:
    """
    matching_n_grams: Counter = Counter()
    for n in hyp_n_grams_counts:
        matching_n_grams[n] = sum(
            [min(ref_n_grams_counts[n][n_gram], hyp_n_grams_counts[n][n_gram]) for n_gram in hyp_n_grams_counts[n]]
        )
    return matching_n_grams


def _sum_over_dict_and_counter(total_n_grams: Dict[int, Tensor], n_grams: Counter) -> Dict[int, Tensor]:
    for n in n_grams:
        total_n_grams[n] += n_grams[n]
    return total_n_grams


def _get_fscore(
    matching_char_n_grams: Counter,
    matching_word_n_grams: Counter,
    ref_char_n_grams: Counter,
    ref_word_n_grams: Counter,
    hyp_char_n_grams: Counter,
    hyp_word_n_grams: Counter,
    n_order: float,
    beta: float,
) -> float:

    def _get_n_gram_fscore(
        matching_n_grams: Counter, ref_n_grams: Counter, hyp_n_grams: Counter, beta: float
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        precision: Dict[int, float] = {
            matching_n_grams[n] / hyp_n_grams[n] if hyp_n_grams[n] > 0 else 1e-16 for n in matching_n_grams
        }
        recall: Dict[int, float] = {
            matching_n_grams[n] / ref_n_grams[n] if ref_n_grams[n] > 0 else 1e-16 for n in matching_n_grams
        }

        denominator: Dict[int, float] = {
            max(beta**2 * precision[n] + recall[n], 1e-16) for n in matching_n_grams
        }
        f_score: Dict[int, float] = {
            (1 + beta**2) * precision[n] * recall[n] / denominator[n] for n in matching_n_grams
        }

        return f_score

    char_n_gram_f_score = _get_n_gram_fscore(
        matching_char_n_grams, ref_char_n_grams, hyp_char_n_grams, beta
    )
    word_n_gram_f_score = _get_n_gram_fscore(
        matching_word_n_grams, ref_word_n_grams, hyp_word_n_grams, beta
    )

    f_score = (sum(char_n_gram_f_score.values()) + sum(word_n_gram_f_score.values())) / n_order
    return f_score


def _sentence_level_chrf_score(
    references: List[str],
    hyp_char_n_grams_counts: Dict[int, Counter],
    hyp_word_n_grams_counts: Dict[int, Counter],
    hyp_char_n_grams: int,
    hyp_word_n_grams: int,
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
):
    best_f_score = 0.0

    for reference in references:
        ref_char_n_grams_counts, ref_word_n_grams_counts, ref_char_n_grams, ref_word_n_grams = (
            _get_n_grams_counts_and_total_ngrams(reference, n_char_order, n_word_order)
        )
        matching_char_n_grams = _get_ngram_matches(ref_char_n_grams_counts, hyp_char_n_grams_counts)
        matching_word_n_grams = _get_ngram_matches(ref_word_n_grams_counts, hyp_word_n_grams_counts)

        f_score = _get_fscore(
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


def _chrf_score_update(
    reference_corpus: List[List[str]],
    hypothesis_corpus: List[str],
    total_ref_char_n_grams: Dict[int, Tensor],
    total_ref_word_n_grams: Dict[int, Tensor],
    total_hyp_char_n_grams: Dict[int, Tensor],
    total_hyp_word_n_grams: Dict[int, Tensor],
    matching_char_n_grams: Dict[int, Tensor],
    matching_word_n_grams: Dict[int, Tensor],
    n_char_order: int,
    n_word_order: int,
    n_order: float,
    beta: float,
    sentence_chrf_score: Optional[List[Tensor]] = None,
):
    for (references, hypothesis) in zip(reference_corpus, hypothesis_corpus):
        hyp_char_n_grams_counts, hyp_word_n_grams_counts, hyp_char_n_grams, hyp_word_n_grams = (
            _get_n_grams_counts_and_total_ngrams(hypothesis, n_char_order, n_word_order)
        )
        total_hyp_char_n_grams = _sum_over_dict_and_counter(total_hyp_char_n_grams, hyp_char_n_grams)
        total_hyp_word_n_grams = _sum_over_dict_and_counter(total_hyp_word_n_grams, hyp_word_n_grams)


def _chrf_score_compute():
    pass


def chrf_score(
    reference_corpus: Union[List[str], List[List[str]]],
    hypothesis_corpus: Union[str, List[str]],
    n_char_order: int = 6,
    n_word_order: int = 2,
    beta: float = 2.0,
    sentence_level_score: bool = False,
):
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]
    
    # Ensure reference corpus is properly of a type List[List[str]]
    if all(isinstance(ref, str) for ref in reference_corpus):
        if len(hypothesis_corpus) == 1:
            reference_corpus = [reference_corpus]  # typing: ignore
        else:
            reference_corpus = [[ref] for ref in reference_corpus]  # typing: ignore

    if len(reference_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")

    n_order = float(n_char_order + n_word_order)

    total_ref_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_char_order}
    total_ref_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_word_order}
    total_hyp_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_char_order}
    total_hyp_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_word_order}

    matching_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_char_order}
    matching_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0, dtype=torch.float) for n in n_word_order}

    sentence_chrf_score: Optional[List[Tensor]] = [] if sentence_level_score else None
