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
from collections.abc import Sequence
from typing import Any, List, Union

from torch import Tensor, stack, tensor

from torchmetrics.functional.text.helper import _validate_inputs
from torchmetrics.utilities.imports import _NLTK_AVAILABLE

__doctest_requires__ = {("meteor_score",): ["nltk"]}


def _ensure_nltk_wordnet_is_downloaded() -> None:
    """Check whether `nltk` `wordnet` is downloaded.

    If not, try to download it if a machine is connected to the internet.

    """
    import nltk

    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True, force=False, halt_on_error=False, raise_on_error=True)
        except ValueError as err:
            raise OSError(
                "`nltk` resource `wordnet` is not available on a disk and cannot be downloaded as a machine is not "
                "connected to the internet."
            ) from err


def _match_enums(
    enum_hypothesis: List[tuple], enum_reference: List[tuple]
) -> tuple[List[tuple], List[tuple], List[tuple]]:
    """Map words that are equal between the enumerated hypothesis and reference and drop them from both lists."""
    word_match = []
    for i in range(len(enum_hypothesis) - 1, -1, -1):
        for j in range(len(enum_reference) - 1, -1, -1):
            if enum_hypothesis[i][1] == enum_reference[j][1]:
                word_match.append((enum_hypothesis[i][0], enum_reference[j][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_hypothesis, enum_reference


def _stem_match(
    enum_hypothesis: List[tuple], enum_reference: List[tuple], stemmer: Any
) -> tuple[List[tuple], List[tuple], List[tuple]]:
    """Match the remaining words on their stem, so that e.g. `ensures` and `ensure` count as a match."""
    stemmed_hypothesis = [(index, stemmer.stem(word)) for index, word in enum_hypothesis]
    stemmed_reference = [(index, stemmer.stem(word)) for index, word in enum_reference]
    return _match_enums(stemmed_hypothesis, stemmed_reference)


def _wordnet_match(
    enum_hypothesis: List[tuple], enum_reference: List[tuple], wordnet: Any
) -> tuple[List[tuple], List[tuple], List[tuple]]:
    """Match the remaining words if a hypothesis word shares a WordNet synonym with a reference word."""
    word_match = []
    for i in range(len(enum_hypothesis) - 1, -1, -1):
        hypothesis_syns = {
            lemma.name()
            for synset in wordnet.synsets(enum_hypothesis[i][1])
            for lemma in synset.lemmas()
            if lemma.name().find("_") < 0
        }
        hypothesis_syns.add(enum_hypothesis[i][1])
        for j in range(len(enum_reference) - 1, -1, -1):
            if enum_reference[j][1] in hypothesis_syns:
                word_match.append((enum_hypothesis[i][0], enum_reference[j][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_hypothesis, enum_reference


def _align_words(hypothesis: List[str], reference: List[str], stemmer: Any, wordnet: Any) -> List[tuple]:
    """Align hypothesis and reference words by exact, then stemmed, then synonym matching."""
    enum_hypothesis = list(enumerate(hypothesis))
    enum_reference = list(enumerate(reference))

    exact, enum_hypothesis, enum_reference = _match_enums(enum_hypothesis, enum_reference)
    stem, enum_hypothesis, enum_reference = _stem_match(enum_hypothesis, enum_reference, stemmer)
    wns, enum_hypothesis, enum_reference = _wordnet_match(enum_hypothesis, enum_reference, wordnet)

    return sorted(exact + stem + wns, key=lambda match: match[0])


def _count_chunks(matches: List[tuple]) -> int:
    """Count the fewest chunks of matched words that are contiguous in both hypothesis and reference."""
    if not matches:
        return 0
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (matches[i + 1][1] == matches[i][1] + 1):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def _single_meteor_score(
    hypothesis: List[str], reference: List[str], stemmer: Any, wordnet: Any, alpha: float, beta: float, gamma: float
) -> float:
    """Compute the METEOR score for a single hypothesis against a single reference."""
    hypothesis_length = len(hypothesis)
    reference_length = len(reference)
    matches = _align_words(hypothesis, reference, stemmer, wordnet)
    matches_count = len(matches)
    if matches_count == 0 or hypothesis_length == 0 or reference_length == 0:
        return 0.0
    precision = matches_count / hypothesis_length
    recall = matches_count / reference_length
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    fragmentation = _count_chunks(matches) / matches_count
    penalty = gamma * fragmentation**beta
    return (1 - penalty) * fmean


def _meteor_score_update(
    preds: Union[str, Sequence[str]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
    stemmer: Any,
    wordnet: Any,
    alpha: float,
    beta: float,
    gamma: float,
) -> List[Tensor]:
    """Compute the sentence level METEOR score for each hypothesis in ``preds``.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        stemmer: A stemmer used to normalise words before matching.
        wordnet: A WordNet corpus reader used for synonym matching.
        alpha: Parameter controlling the relative weight of precision and recall.
        beta: Parameter controlling the shape of the fragmentation penalty.
        gamma: Relative weight assigned to the fragmentation penalty.

    Return:
        A list of sentence level METEOR scores.

    """
    target, preds = _validate_inputs(target, preds)

    results = []
    for pred, references in zip(preds, target):
        pred_tokens = pred.lower().split()
        score = max(
            _single_meteor_score(pred_tokens, ref.lower().split(), stemmer, wordnet, alpha, beta, gamma)
            for ref in references
        )
        results.append(tensor(score))
    return results


def _meteor_score_compute(sentence_results: List[Tensor]) -> Tensor:
    """Average the sentence level METEOR scores into a corpus level score."""
    if not sentence_results:
        return tensor(0.0)
    return stack(sentence_results).mean()


def meteor_score(
    preds: Union[str, Sequence[str]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Calculate `METEOR`_ score of machine translated text with one or more references.

    METEOR aligns the hypothesis to each reference using exact, stemmed and WordNet synonym matches, scores the
    alignment with a recall weighted harmonic mean of precision and recall, and applies a fragmentation penalty. When
    several references are given the best scoring reference is used, and the corpus level score is the average over all
    hypotheses.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        alpha: Parameter controlling the relative weight of precision and recall.
        beta: Parameter controlling the shape of the fragmentation penalty.
        gamma: Relative weight assigned to the fragmentation penalty.
        return_sentence_level_score: An indication whether a sentence level METEOR score to be returned.

    Return:
        A corpus level METEOR score. If ``return_sentence_level_score=True`` a list of sentence level scores is also
        returned.

    Raises:
        ModuleNotFoundError:
            If ``nltk`` package is not installed.

    Example:
        >>> from torchmetrics.functional.text import meteor_score
        >>> preds = ['the cat sat on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> meteor_score(preds, target)
        tensor(0.6250)

    """
    if not _NLTK_AVAILABLE:
        raise ModuleNotFoundError("METEOR metric requires that `nltk` is installed. Use `pip install nltk`.")
    import nltk

    _ensure_nltk_wordnet_is_downloaded()

    stemmer = nltk.stem.porter.PorterStemmer()

    sentence_results = _meteor_score_update(preds, target, stemmer, nltk.corpus.wordnet, alpha, beta, gamma)
    average = _meteor_score_compute(sentence_results)
    if return_sentence_level_score:
        return average, stack(sentence_results) if sentence_results else tensor([])
    return average
