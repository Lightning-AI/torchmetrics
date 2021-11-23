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

##############

# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Uday Krishna <udaykrishna5@gmail.com>
# Contributor: Tom Aarsen
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from itertools import chain
from typing import Any, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _NLTK_CORPUS_AVAILABLE

if _NLTK_AVAILABLE and _NLTK_CORPUS_AVAILABLE:
    from nltk.corpus import wordnet as nltk_wordnet
    from nltk.stem import SnowballStemmer
else:
    PorterStemmer, StemmerI, WordNetCorpusReader, nltk_wordnet = None, None, None, None


_SUPPORTED_STEMMER_LANGUAGES = (
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "hungarian",
    "italian",
    "norwegian",
    "porter",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
)

_SUPPORTED_STEMMER_LANGUAGES_TYPE = Literal[
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "hungarian",
    "italian",
    "norwegian",
    "porter",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
]


def _get_function_words(lang: str) -> Sequence[str]:
    function_words_lang = "english" if lang == "porter" else lang
    from torchmetrics.functional.text.resources import function_words as function_words_resource

    if hasattr(function_words_resource, function_words_lang):
        function_words = getattr(function_words_resource, function_words_lang).FUNCTION_WORDS
    else:
        function_words = function_words_resource.other.FUNCTION_WORDS
    return function_words


class _METEORScoreComponents(NamedTuple):
    content_matches_count: Tensor
    function_matches_count: Tensor
    content_reference_len: Tensor
    function_reference_len: Tensor
    content_hypothesis_len: Tensor
    function_hypothesis_len: Tensor
    frag_frac: Tensor


class _NLTKStemmerWrapper:
    """TorchMetrics wrapper for `nltk` stemmers."""

    def __init__(self, stemmer_lang: _SUPPORTED_STEMMER_LANGUAGES_TYPE = "porter") -> None:
        """
        Args:
            stemmer:
                A name of stemmer from `nltk` package to be used.

        Raises:
            KeyError:
                If invalid stemmer class is chosen.
        """
        self.stemmer_lang = stemmer_lang

    def __call__(self, word: str) -> str:
        """Return a stemmed word.

        Args:
            word:
                A word to be stemmed.

        Returns:
            A stemmed word.
        """
        if SnowballStemmer is not None:
            stemmer = SnowballStemmer(self.stemmer_lang)
            return stemmer.stem(word)
        # To comply with mypy typing on several places
        return word


class _NLTKWordnetWrapper:
    """TorchMetrics wrapper for `nltk` wordnet corpuses."""

    def __call__(self, word: str) -> Optional[Any]:
        """
        Args:
            word:
                An input word.

        Returns:
            A set of synonyms.
        """
        if nltk_wordnet is not None:
            return nltk_wordnet.synsets(word)
        # To comply with mypy typing on several places
        return None


def _generate_synonyms(word: str, wordnet: _NLTKWordnetWrapper) -> Set[str]:
    """Generate synonyms using `nltk` wordnet corpus for an input word.

    Args:
        word:
            An input word.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.

    Returns:
        A set of synonyms for the input word.
    """
    synsets = wordnet(word)
    if synsets is not None:
        synonyms_set = set(
            chain.from_iterable(
                (lemma.name() for lemma in synset.lemmas() if lemma.name().find("_") < 0) for synset in synsets
            )
        ).union({word})
    else:
        synonyms_set = {word}
    return synonyms_set


def _match_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]], function_words: Sequence[str]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
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
    content_word_match = []
    function_word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        for j in range(len(enum_reference))[::-1]:
            if enum_hypothesis[i][1] == enum_reference[j][1]:
                if enum_hypothesis[i][1] in function_words:
                    function_word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                else:
                    content_word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return content_word_match, function_word_match, enum_reference, enum_hypothesis


def _match_stem_enums(
    enum_reference: List[Tuple[int, str]],
    enum_hypothesis: List[Tuple[int, str]],
    stemmer: _NLTKStemmerWrapper,
    function_words: Sequence[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Stem each word in both reference and hypothesis and then aligns/matches stemmed words in the hypothesis to
    the reference.

    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothessis.
        stemmer:
            `_NLTKWordnetWrapper` object utilizing `nltk` stemmer.
        function_words:
            A tuple of words considered to be function ones in a target language.

    Returns:
        A tuple of lists:
            A list of tuples mapping matched stemmed reference and hypothesis words.
            An enumerated list of unmatched stemmed reference words.
            An enumerated list of unmatched stemmed hypothesis words.
    """
    stemmed_enum_reference = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_reference]
    stemmed_enum_hypothesis = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_hypothesis]
    return _match_enums(stemmed_enum_reference, stemmed_enum_hypothesis, function_words)


def _match_synonym_enums(
    enum_reference: List[Tuple[int, str]],
    enum_hypothesis: List[Tuple[int, str]],
    wordnet: _NLTKWordnetWrapper,
    function_words: Sequence[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Args:
        enum_reference:
            An enumerated list of a tokenized reference.
        enum_hypothesis:
            An enumerated list of a tokenized hypothesis.
        wordnet:
            `_NLTKWordnetWrapper` object utilizing `nltk` wordnet corpus for looking up for synonyms.
        function_words:
            A tuple of words considered to be function ones in a target language.

    Returns:
        A tuple of lists:
            A list of tuples mapping matched reference and hypothesis words based on synonym match..
            An enumerated list of unmatched stemmed reference words.
            An enumerated list of unmatched stemmed hypothesis words.
    """
    content_word_match = []
    function_word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        hypothesis_synonyms = _generate_synonyms(enum_hypothesis[i][1], wordnet)
        for j in range(len(enum_reference))[::-1]:
            if enum_reference[j][1] in hypothesis_synonyms:
                if enum_reference[j][1] in function_words:
                    function_word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                else:
                    content_word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return content_word_match, function_word_match, enum_reference, enum_hypothesis


def _align_enum_words(
    enum_reference: List[Tuple[int, str]],
    enum_hypothesis: List[Tuple[int, str]],
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
    function_words: Sequence[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, float]:
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
        function_words:
            A tuple of words considered to be function ones in a target language.

    Returns:
        An enumerated sorted list of matched words.
        A length of the list of matched words.
    """
    exact_content_matches, exact_function_matches, enum_reference, enum_hypothesis = _match_enums(
        enum_reference, enum_hypothesis, function_words
    )
    stem_content_matches, stem_function_matches, enum_reference, enum_hypothesis = _match_stem_enums(
        enum_reference, enum_hypothesis, stemmer, function_words
    )
    synonym_content_matches, synonym_function_matches, enum_reference, enum_hypothesis = _match_synonym_enums(
        enum_reference, enum_hypothesis, wordnet, function_words
    )

    sorted_content_matches = sorted(
        exact_content_matches + stem_content_matches + synonym_content_matches, key=lambda wordpair: wordpair[0]
    )
    sorted_function_matches = sorted(
        exact_function_matches + stem_function_matches + synonym_function_matches, key=lambda wordpair: wordpair[0]
    )
    return (
        sorted_content_matches,
        sorted_function_matches,
        float(len(sorted_content_matches)),
        float(len(sorted_function_matches)),
    )


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
    functions_words: Sequence[str],
) -> _METEORScoreComponents:
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
        function_words:
            A tuple of words considered to be function ones in a target language.

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
    function_reference_len = float(len([ref[1] for ref in enum_reference if ref[1] in functions_words]))
    content_reference_len = float(len(enum_reference) - function_reference_len)
    function_hypothesis_len = float(len([hyp[1] for hyp in enum_hypothesis if hyp[1] in functions_words]))
    content_hypothesis_len = float(len(enum_hypothesis) - function_hypothesis_len)

    content_matches, function_matches, content_matches_count, function_matches_count = _align_enum_words(
        enum_reference, enum_hypothesis, stemmer, wordnet, functions_words
    )
    matches_count = content_matches_count + function_matches_count
    matches = sorted(content_matches + function_matches, key=lambda wordpair: wordpair[0])

    frag_frac = _count_chunks(matches) / matches_count if matches_count != 0 else 0.0
    return _METEORScoreComponents(
        content_matches_count=tensor(content_matches_count),
        function_matches_count=tensor(function_matches_count),
        content_reference_len=tensor(content_reference_len),
        function_reference_len=tensor(function_reference_len),
        content_hypothesis_len=tensor(content_hypothesis_len),
        function_hypothesis_len=tensor(function_hypothesis_len),
        frag_frac=tensor(frag_frac),
    )


def _calculate_meteor_score(
    content_matches_count: Tensor,
    function_matches_count: Tensor,
    content_reference_len: Tensor,
    function_reference_len: Tensor,
    content_hypothesis_len: Tensor,
    function_hypothesis_len: Tensor,
    frag_frac: Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
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
            Expected `alpha` to be between 0 and 1.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
            Expected `beta` be greater than or equal to 0.
        gamma:
            A relative weight assigned to fragmentation penalty.
            Expected `gamma` to be between 0 and 1.
        delta:
            A relative weight to content and function words. Relevant only if `use_function_words = True`.
            Expected `delta` to be between 0 and 1.

    Returns:
        Sentence-level METEOR score for a given reference and hypothesis.
    """
    try:
        precision = (delta * content_matches_count + (1 - delta) * function_matches_count) / (
            delta * content_hypothesis_len + (1 - delta) * function_hypothesis_len
        )
        recall = (delta * content_matches_count + (1 - delta) * function_matches_count) / (
            delta * content_reference_len + (1 - delta) * function_reference_len
        )
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    except ZeroDivisionError:
        return tensor(0.0)
    penalty = gamma * frag_frac ** beta
    meteor_score = (1.0 - penalty) * fmean
    return meteor_score


def _meteor_score_update(
    reference_corpus: Union[List[str], List[List[str]]],
    hypothesis_corpus: Union[str, List[str]],
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
    function_words: Sequence[str],
) -> List[Tuple[_METEORScoreComponents, ...]]:
    """
    Args:
        reference_corpus:
            An iterable of iterables of reference corpus. Either a list of reference corpora or a list of lists of
            reference corpora.
        hypothesis_corpus:
            An iterable of machine translated corpus. Either a single hypothesis corpus or a list of hypothesis
            corpora.
        stemmer:
            `_NLTKStemmerWrapper` object
        wordnet:
            `_NLTKWordnetWrapper` object
        function_words:
            A tuple of words considered to be function ones in a target language.

    Returns:
        Individual components for sentence-level METEOR score for given reference and hypothesis corpora

    Raises:
        ValueError:
            If length of reference and hypothesis corpus differs.
    """
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]

    # Check if reference_corpus is a type of List[str]
    if all(isinstance(ref, str) for ref in reference_corpus):
        if len(hypothesis_corpus) == 1:
            reference_corpus = [reference_corpus]  # type: ignore
        else:
            reference_corpus = [[reference] for reference in reference_corpus]  # type: ignore

    if len(reference_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")

    results: List[Tuple[_METEORScoreComponents, ...]] = []
    for references, hypothesis in zip(reference_corpus, hypothesis_corpus):
        results.append(
            tuple(
                _calculate_meteor_components(reference, hypothesis, stemmer, wordnet, function_words)
                for reference in references
            )
        )
    return results


def _meteor_score_compute(
    meteor_score_components: List[Tuple[_METEORScoreComponents, ...]],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Tensor:
    """
    Args:
        meteor_score_components:
            A named tuple containing `matches_count`, `reference_len`, `hypothesis_len` and `frag_frac` used for the
            calculation of the METEOR score.
        alpha:
            A parameter for controlling relative weights of precision and recall.
            Expected `alpha` to be between 0 and 1.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
            Expected `beta` be greater than or equal to 0.
        gamma:
            A relative weight assigned to fragmentation penalty.
            Expected `gamma` to be between 0 and 1.
        delta:
            A relative weight to content and function words. Relevant only if `use_function_words = True`.
            Expected `delta` to be between 0 and 1.

    Returns:
        Sentence-level METEOR score for given references and a single hypothesis
    """
    # Sentence-level METEOR score
    sentence_results: List[Tensor] = []
    for components in meteor_score_components:
        sentence_results.append(
            max(  # type: ignore
                _calculate_meteor_score(
                    sentence_pair_components.content_matches_count,
                    sentence_pair_components.function_matches_count,
                    sentence_pair_components.content_reference_len,
                    sentence_pair_components.function_reference_len,
                    sentence_pair_components.content_hypothesis_len,
                    sentence_pair_components.function_hypothesis_len,
                    sentence_pair_components.frag_frac,
                    alpha,
                    beta,
                    gamma,
                    delta,
                )
                for sentence_pair_components in components
            )
        )
    # TODO: Add Corpus-level METEOR score with an update to METEOR v1.5

    return tensor(sentence_results)


def meteor_score(
    reference_corpus: Union[List[str], List[List[str]]],
    hypothesis_corpus: Union[str, List[str]],
    lang: _SUPPORTED_STEMMER_LANGUAGES_TYPE = "porter",
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    delta: float = 0.75,
    use_function_words: bool = False,
) -> Tensor:
    """Calculate `METEOR Score`_ used for automatic machine translation evaluation.

    Current implementation follows the METEOR metric orignially proposed in the paper of Lavie and Agarwal [1], whicg
    is implemented in `nltk` package. It is itended to upgrade this metric to the currently used version 1.5 [2].

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus. Either a list of reference corpora or a list of lists of
            reference corpora.
        hypothesis_corpus:
            An iterable of machine translated corpus. Either a single hypothesis corpus or a list of hypothesis
            corpora.
        stemmer:
            A name of stemmer from `nltk` package to be used.
        alpha:
            A parameter for controlling relative weights of precision and recall.
            Expected `alpha` to be between 0 and 1.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
            Expected `beta` be greater than or equal to 0.
        gamma:
            A relative weight assigned to fragmentation penalty.
            Expected `gamma` to be between 0 and 1.
        delta:
            A relative weight to content and function words. Relevant only if `use_function_words = True`.
            Expected `delta` to be between 0 and 1.
        use_function_words:
            An indication whether to discriminate between content and function words. A list of function words is
            derived on monolingual corpora. All words with relative frequency above 10^{-3} are considered to be
            function words.


    Returns:
        Tensor with sentence-level METEOR score.

    Raises:
        ValueError:
            If `nltk` package is not installed.
        ValueError:
            If `alpha` is not between 0 and 1.
        ValueError:
            If `beta` is not greater than or equal to 0.
        ValueError:
            If `gamma` is not between 0 and 1.

    Example:
        >>> import nltk
        >>> nltk.download('wordnet')
        True
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

    if not _NLTK_AVAILABLE:
        raise ValueError(
            "METEOR metric requires that nltk is installed. Use `pip install nltk` or `pip install torchmetrics[text].`"
        )
    if not _NLTK_CORPUS_AVAILABLE:
        raise ValueError(
            "METEOR metric requires that nltk wordnet corpus is downloaded. Use `python -m nltk.downloader wordnet`."
        )

    if lang not in _SUPPORTED_STEMMER_LANGUAGES:
        raise ValueError(f"`stemmer_lang` is expected to be chosen from {_SUPPORTED_STEMMER_LANGUAGES}.")

    if not 0 <= alpha <= 1:
        raise ValueError("Expected `alpha` argument to be between 0 and 1.")
    if beta < 0:
        raise ValueError("Expected `beta` argument to be greater than or equal to 0.")
    if not 0 <= gamma <= 1:
        raise ValueError("Expected `gamma` argument to be between 0 and 1.")
    if not 0 <= delta <= 1:
        raise ValueError("Expected `delta` argument to be between 0 and 1.")

    stemmer_class = _NLTKStemmerWrapper(lang)
    wordnet_class = _NLTKWordnetWrapper()

    if use_function_words:
        function_words = _get_function_words(lang)
    else:
        function_words = ()

    meteor_score_components = _meteor_score_update(
        reference_corpus, hypothesis_corpus, stemmer_class, wordnet_class, function_words
    )
    return _meteor_score_compute(meteor_score_components, alpha, beta, gamma, delta)
