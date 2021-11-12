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
# Authors: torchtext authors
# Date: 2021-11-15
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#meteor_score

##############

# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Uday Krishna <udaykrishna5@gmail.com>
# Contributor: Tom Aarsen
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import warnings
from itertools import chain
from typing import Any, Dict, List, Set, Tuple, Union

from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _NLTK_AVAILABLE


class _NLTKStemmerWrapper:
    """TorchMetrics wrapper for `nltk` stemmers."""

    _STEMMER_CLASS = {"porter": "PorterStemmer"}

    def __init__(self, stemmer: Literal["porter"] = "porter", *args: Any, **kwargs: Any) -> None:
        """
        Args:
            stemmer:

        Raises:
        """
        if not _NLTK_AVAILABLE:
            raise ValueError("Stemmer requires that nltk is installed. Use `pip install nltk`.")
        from nltk import stem

        try:
            stemmer_class = getattr(getattr(stem, stemmer), self._STEMMER_CLASS[stemmer])
        except ImportError:
            print(123)

        self.stemmer = stemmer_class(*args, **kwargs)

    def __call__(self, word: str) -> str:
        """
        Args:
            word:

        Returns:
        """
        return self.stemmer.stem(word)


class _NLTKWordnetWrapper:
    """TorchMetrics wrapper for `nltk` wordnet corpuses."""

    _WORDNET_CLASS = {"wordnet": "wordnet"}

    def __init__(self, wordnet: Literal["wordnet"], *args: Any, **kwargs: Any) -> None:
        """
        Args:
            wordnet:

        Raises:
        """
        if not _NLTK_AVAILABLE:
            raise ValueError("Stemmer requires that nltk is installed. Use `pip install nltk`.")
        from nltk import corpus

        try:
            self.wordnet = getattr(corpus, self._WORDNET_CLASS[wordnet])
        except ImportError:
            print(123)

    def __call__(self, word: str):
        """
        Args:
            word:

        Returns:
        """
        return self.wordnet.synsets(word)


def _generate_synonyms(word: str, wordnet: _NLTKWordnetWrapper) -> Set[str]:
    """Generate synonyms using `nltk` wordnet corpus for an input word.

    Args:
        word:
            An input word.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.

    Returns:
        A set of found synonyms for the input word.
    """
    synonyms_set = set(
        chain.from_iterable(
            (lemma.name() for lemma in synset.lemmas() if lemma.name().find("_") < 0) for synset in wordnet(word)
        )
    ).union({word})
    return synonyms_set


def _match_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Align/match words in the hypothesis to the reference.

    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothessis.

    Returns:
        A tuple of lists:
            A list of tuples mapping matched reference and hypothesis words.
            An enumerated list of unmatched reference words.
            An enumerated list of unmatched hypothesis words.
    """
    word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        for j in range(len(enum_reference))[::-1]:
            if enum_hypothesis[i][1] == enum_reference[j][1]:
                word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_reference, enum_hypothesis


def _match_stem_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]], stemmer: _NLTKStemmerWrapper
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Stem each word in both reference and hypothesis and then aligns/matches stemmed words in the hypothesis to
    the reference.

    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothessis.
        stemmer:
            `_NLTKWordnetWrapper` object utilizing `nltk` stemmer.

    Returns:
        A tuple of lists:
            A list of tuples mapping matched stemmed reference and hypothesis words.
            An enumerated list of unmatched stemmed reference words.
            An enumerated list of unmatched stemmed hypothesis words.
    """
    stemmed_enum_reference = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_reference]
    stemmed_enum_hypothesis = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_hypothesis]
    return _match_enums(stemmed_enum_reference, stemmed_enum_hypothesis)


def _match_synonym_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]], wordnet: _NLTKWordnetWrapper
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothesis.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.

    Returns:
        A tuple of lists:
            A list of tuples mapping matched reference and hypothesis words based on synonym match..
            An enumerated list of unmatched stemmed reference words.
            An enumerated list of unmatched stemmed hypothesis words.
    """
    word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        hypothesis_synonyms = _generate_synonyms(enum_hypothesis[i][1], wordnet)
        for j in range(len(enum_reference))[::-1]:
            if enum_reference[j][1] in hypothesis_synonyms:
                word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_reference, enum_hypothesis


def _align_enum_words(
    enum_reference: List[str], enum_hypothesis: List[str], stemmer: _NLTKStemmerWrapper, wordnet: _NLTKWordnetWrapper
) -> List[Tuple[int, int]]:
    """Align/match words in the hypothesis to the reference. This is achieved by sequentially applying exact match,
    stemmed match and synonym match based on `nltk` wordnet.

    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothesis.
        stemmer:
            `_NLTKWordnetWrapper` object utilizing `nltk` stemmer.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.

    Returns:
        An enumerated sorted list of matched words.
        A length of the list of matched words.
    """
    exact_matches, enum_reference, enum_hypothesis = _match_enums(enum_reference, enum_hypothesis)
    stem_matches, enum_reference, enum_hypothesis = _match_stem_enums(enum_reference, enum_hypothesis, stemmer)
    synonym_matches, enum_reference, enum_hypothesis = _match_synonym_enums(enum_reference, enum_hypothesis, wordnet)

    sorted_matches = sorted(exact_matches + stem_matches + synonym_matches, key=lambda wordpair: wordpair[0])
    return sorted_matches, float(len(sorted_matches))


def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    """Count the fewest possible number of chunks such that matched unigrams of each chunk are adjacent to each
    other. This is used to calculate the fragmentation part of the metric.

    Args:
        matches:
            A list of a mapping of matched reference and hypothesis words.

    Returns:
        A number of chunks a sentence is divided into post alignment.
    """
    chunks = 1
    chunks += sum(
        1
        for i in range(len(matches) - 1)
        if not ((matches[i + 1][0] == matches[i][0] + 1) and (matches[i + 1][1] == matches[i][1] + 1))
    )
    return chunks


def _calculate_meteor_components(
    reference: str,
    hypothesis: str,
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
) -> Dict[str, Tensor]:
    """Calculate components used for the METEOR score calculation.

    Args:
        reference:
            A single reference sentence.
        hypothesis:
            A single hypothesis sentence.
        stemmer:
            `_NLTKWordnetWrapper` object utilizing `nltk` stemmer.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.

    Returns:
        A python dictionary containing components for calculating METEOR score.
            matches_count:
                A number of matches between reference and hypothesis sentences.
            reference_len:
                A length of a tokenized reference sentence.
            hypothesis_len:
                A length of a tokenized hypothesis sentence.
            frag_frac:
                A value used for constructing `penalty` component for calculating the METEOR score from F-mean.
    """
    enum_reference = list(enumerate(reference.split()))
    enum_hypothesis = list(enumerate(hypothesis.split()))
    reference_len = float(len(enum_reference))
    hypothesis_len = float(len(enum_hypothesis))
    matches, matches_count = _align_enum_words(enum_reference, enum_hypothesis, stemmer, wordnet)
    frag_frac = _count_chunks(matches) / matches_count if matches_count != 0 else 0.0
    return {
        "matches_count": tensor(matches_count),
        "reference_len": tensor(reference_len),
        "hypothesis_len": tensor(hypothesis_len),
        "frag_frac": tensor(frag_frac),
    }


def _calculate_meteor_score(
    matches_count: Tensor,
    reference_len: Tensor,
    hypothesis_len: Tensor,
    frag_frac: Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tensor:
    """Calculate METEOR score using pre-calculated components.

    Args:
        matches_count:
            A number of matches between reference and hypothesis sentences.
        reference_len:
            A length of a tokenized reference sentence.
        hypothesis_len:
            A length of a tokenized hypothesis sentence.
        frag_frac:
            A value used for constructing `penalty` component for calculating the METEOR score from F-mean.
        alpha:
            A parameter for controlling relative weights of precision and recall.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
        gamma:
            A relative weight assigned to fragmentation penalty.

    Returns:
        Sentence-level METEOR score for a given reference and hypothesis.
    """
    try:
        precision = matches_count / hypothesis_len
        recall = matches_count / reference_len
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    except ZeroDivisionError:
        return tensor(0.0)
    penalty = gamma * frag_frac ** beta
    meteor_score = (1.0 - penalty) * fmean
    return meteor_score


def _meteor_score_update(
    reference_corpus: List[List[str]],
    hypothesis_corpus: List[str],
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
) -> List[List[Dict[str, Tensor]]]:
    """
    Args:
        reference_corpus:
            An iterable of iterables of reference corpus.
        hypothesis_corpus:
            An iterable of machine translated corpus.
        stemmer:
            `_NLTKStemmerWrapper` object
        wordnet:
            `_NLTKWordnetWrapper` object
        alpha:
            A parameter for controlling relative weights of precision and recall.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
        gamma:
            A relative weight assigned to fragmentation penalty.

    Return:
        Sentence-level METEOR score for given reference and hypothesis corpora
    """

    results: List[List[Dict[str, Tensor]]] = []
    for references, hypothesis in zip(reference_corpus, hypothesis_corpus):
        results.append(
            [_calculate_meteor_components(reference, hypothesis, stemmer, wordnet) for reference in references]
        )
    return results


def _meteor_score_compute(
    meteor_score_components: List[List[Dict[str, Tensor]]], alpha: float, beta: float, gamma: float
) -> Tensor:
    """
    Args:
        meteor_score_components:

        alpha:
            A parameter for controlling relative weights of precision and recall.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
        gamma:
            A relative weight assigned to fragmentation penalty.

    Returns:

    """
    # Sentence-level METEOR score
    sentence_results: List[Tensor] = []
    for components in meteor_score_components:
        sentence_results.append(
            max(
                _calculate_meteor_score(**sentence_pair_components, alpha=alpha, beta=beta, gamma=gamma)
                for sentence_pair_components in components
            )
        )
    return tensor(sentence_results)


def meteor_score(
    reference_corpus: Union[List[str], List[List[str]]],
    hypothesis_corpus: Union[str, List[str]],
    stemmer: Literal["porter"] = "porter",
    wordnet: Literal["wordnet"] = "wordnet",
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> Tensor:
    """Calculate `METEOR Score`_ used for automatic machine translation evaluation.

    Current implementation follows the METEOR metric orignially proposed in the paper of Lavie and Agarwal [1], whicg
    is implemented in `nltk` package. It is itended to upgrade this metric to the currently used version 1.5 [2].

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus.
        hypothesis_corpus:
            An iterable of machine translated corpus.
        stemmer:
            A name of stemmer from `nltk` package to be used.
        wordnet:
            A name of wordnet corpus from `nltk` package to be used.
        alpha:
            A parameter for controlling relative weights of precision and recall.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
        gamma:
            A relative weight assigned to fragmentation penalty.

    Returns:
        Tensor with sentence-level METEOR score.

    Raises:
        ValueError:
            If `nltk` package is not installed.
        ValueError:
            If length of reference and hypothesis corpus differs.

    Example:
        >>> predictions = ['the cat is on the mat']
        >>> references = [['there is a cat on the mat']]
        >>> meteor_score(references, predictions)  # doctest: +SKIP
        tensor([0.6464])

    References:
    [1] METEOR: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments by Alon
    Lavie and Abhaya Agarwal.

    [2] Meteor Universal: Language Specific Translation Evaluation for Any Target Language by Michael Denkowski and
    Alon Lavie.
    """
    warnings.warn(
        "Current implementation follows the original METEOR metric and thus is not suitable for reporting results "
        "in research papers."
    )
    if not _NLTK_AVAILABLE:
        raise ValueError(
            "METEOR metric requires that nltk is installed. Use `pip install nltk` or `pip install torchmetrics[text].`"
        )

    if len(reference_corpus) > 0 and isinstance(reference_corpus[0], str):
        reference_corpus = [[reference] for reference in reference_corpus]
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]

    if len(reference_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")

    stemmer_class = _NLTKStemmerWrapper(stemmer)
    wordnet_class = _NLTKWordnetWrapper(wordnet)

    meteor_score_components = _meteor_score_update(reference_corpus, hypothesis_corpus, stemmer_class, wordnet_class)
    return _meteor_score_compute(meteor_score_components, alpha, beta, gamma)
