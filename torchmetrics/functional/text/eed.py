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
# Date: 2021-12-07
# Link:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The RWTH Extended Edit Distance (EED) License

# Copyright (c) 2019, RWTH.
# All rights reserved.

# This license is derived from the Q Public License v1.0 and the Qt Non-Commercial License v1.0 which are both Copyright
# by Trolltech AS, Norway. The aim of this license is to lay down the conditions enabling you to use, modify and
# circulate the SOFTWARE, use of third-party application programs based on the Software and publication of results
# obtained through the use of modified and unmodified versions of the SOFTWARE. However, RWTH remain the authors of the
# SOFTWARE and so retain property rights and the use of all ancillary rights. The SOFTWARE is defined as all successive
# versions of EED software and their documentation that have been developed by RWTH.
#
# When you access and use the SOFTWARE, you are presumed to be aware of and to have accepted all the rights and
# obligations of the present license:
#
#  1. You are granted the non-exclusive rights set forth in this license provided you agree to and comply with any all
#     conditions in this license. Whole or partial distribution of the Software, or software items that link with the
#     Software, in any form signifies acceptance of this license for non-commercial use only.
#  2. You may copy and distribute the Software in unmodified form provided that the entire package, including - but not
#     restricted to - copyright, trademark notices and disclaimers, as released by the initial developer of the
#     Software, is distributed.
#  3. You may make modifications to the Software and distribute your modifications, in a form that is separate from the
#     Software, such as patches. The following restrictions apply to modifications:
#     a. Modifications must not alter or remove any copyright notices in the Software.
#     b When modifications to the Software are released under this license, a non-exclusive royalty-free right is
#       granted to the initial developer of the Software to distribute your modification in future versions of the
#       Software provided such versions remain available under these terms in addition to any other license(s) of the
#       initial developer.
#  4. You may distribute machine-executable forms of the Software or machine-executable forms of modified versions of
#     the Software, provided that you meet these restrictions:
#     a. You must include this license document in the distribution.
#     b. You must ensure that all recipients of the machine-executable forms are also able to receive the complete
#        machine-readable source code to the distributed Software, including all modifications, without any charge
#        beyond the costs of data transfer, and place prominent notices in the distribution explaining this.
#     c. You must ensure that all modifications included in the machine-executable forms are available under the terms
#        of this license.
#  5. You may use the original or modified versions of the Software to compile, link and run application programs
#     legally developed by you or by others.
#  6. You may develop application programs, reusable components and other software items, in a non-commercial setting,
#     that link with the original or modified versions of the Software. These items, when distributed, are subject to
#     the following requirements:
#     a. You must ensure that all recipients of machine-executable forms of these items are also able to receive and use
#        the complete machine-readable source code to the items without any charge beyond the costs of data transfer.
#     b. You must explicitly license all recipients of your items to use and re-distribute original and modified
#        versions of the items in both machine-executable and source code forms. The recipients must be able to do so
#        without any charges whatsoever, and they must be able to re-distribute to anyone they choose.
#     c. If an application program gives you access to functionality of the Software for development of application
#        programs, reusable components or other software components (e.g. an application that is a scripting wrapper),
#        usage of the application program is considered to be usage of the Software and is thus bound by this license.
#     d. If the items are not available to the general public, and the initial developer of the Software requests a copy
#        of the items, then you must supply one.
#  7. Users must cite the authors of the Software upon publication of results obtained through the use of original or
#     modified versions of the Software by referring to the following publication:
#     P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”, submitted to WMT
#     2019.
#  8. In no event shall the initial developers or copyright holders be liable for any damages whatsoever, including -
#     but not restricted to - lost revenue or profits or other direct, indirect, special, incidental or consequential
#     damages, even if they have been advised of the possibility of such damages, except to the extent invariable law,
#     if any, provides otherwise.
#  9. You assume all risks concerning the quality or the effects of the SOFTWARE and its use. If the SOFTWARE is
#     defective, you will bear the costs of all required services, corrections or repairs.
#  10. This license has the binding value of a contract.
#  11. The present license and its effects are subject to German law and the competent German Courts.
#
# The Software and this license document are provided "AS IS" with NO EXPLICIT OR IMPLICIT WARRANTY OF ANY KIND,
# INCLUDING WARRANTY OF DESIGN, ADAPTION, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union

from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.text.helper import _validate_inputs


def _distance_between_words(reference_word: str, hypothesis_word: str) -> int:
    """Distance measure used for substitutions/identity operation. Code adapted from
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        reference_word: reference word string
        hypothesis_word: hypothesis word string

    Returns:
        0 for match, 1 for no match
    """
    return int(reference_word != hypothesis_word)


def _eed_function(
    ref: str,
    hyp: str,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> float:
    """Computes extended edit distance score for two lists of strings: hyp and ref. Code adapted from:
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        ref:
            reference string
        hyp:
            hypothesis string
        alpha:
            optimal jump penalty, penalty for jumps between characters
        rho:
            coverage cost, penalty for repetition of characters
        deletion:
            penalty for deletion of character
        insertion:
            penalty for insertion or substitution of character

    Returns:
        Extended edit distance score as float
    """
    number_of_visits = [-1] * (len(hyp) + 1)

    # row[i] stores cost of cheapest path from (0,0) to (i,l) in CDER aligment grid.
    row = [1.0] * (len(hyp) + 1)

    row[0] = 0.0  # CDER initialisation 0,0 = 0.0, rest 1.0
    next_row = [inf] * (len(hyp) + 1)

    for w in range(1, len(ref) + 1):
        for i in range(0, len(hyp) + 1):

            if i > 0:
                next_row[i] = min(
                    next_row[i - 1] + deletion,
                    row[i - 1] + _distance_between_words(ref[w - 1], hyp[i - 1]),
                    row[i] + insertion,
                )
            else:
                next_row[i] = row[i] + 1.0

        min_index = next_row.index(min(next_row))
        number_of_visits[min_index] += 1

        # Long Jumps
        if ref[w - 1] == " ":
            jump = alpha + next_row[min_index]
            next_row = [min(x, jump) for x in next_row]

        row = next_row
        next_row = [inf] * (len(hyp) + 1)

    coverage = rho * sum(x if x >= 0 else 1 for x in number_of_visits)

    return min(1, (row[-1] + coverage) / (float(len(ref)) + coverage))


def _preprocess_en(sentence: str) -> str:
    """Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.
    """
    if not isinstance(sentence, str):
        raise ValueError(f"Only strings allowed during preprocessing step, found {type(sentence)} instead")

    sentence = sentence.rstrip()  # trailing space, tab, or newline

    # Add space before interpunctions
    rules_interpunction = [
        (".", " ."),
        ("!", " !"),
        ("?", " ?"),
        (",", " ,"),
    ]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)

    rules_re = [
        (r"\s+", r" "),  # get rid of extra spaces
        (r"(\d) ([.,]) (\d)", r"\1\2\3"),  # 0 . 1 -> 0.1
        (r"(Dr|Jr|Prof|Rev|Gen|Mr|Mt|Mrs|Ms) .", r"\1."),  # Mr . -> Mr.
    ]
    for pattern, replacement in rules_re:
        sentence = re.sub(pattern, replacement, sentence)

    # Add space between abbreviations
    rules_interpunction = [
        ("e . g .", "e.g."),
        ("i . e .", "i.e."),
        ("U . S .", "U.S."),
    ]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)

    # add space to beginning of string
    sentence = " " + sentence

    return sentence


def _preprocess_ja(sentence: str) -> str:
    """Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.
    """
    if not isinstance(sentence, str):
        raise ValueError(f"Only strings allowed during preprocessing step, found {type(sentence)} instead")

    sentence = sentence.rstrip()  # trailing space, tab, newline
    # characters which look identical actually are identical
    sentence = unicodedata.normalize("NFKC", sentence)
    return sentence


def _eed_compute(scores: Tensor, total_num_sentences: Tensor) -> Tensor:
    """Final step in extended edit distance.

    Args:
        scores: sum of individual sentence scores as a tensor
        total_num_sentences: number of sentences as a tensor

    Returns:
        average of scores as a tensor
    """
    if scores == tensor(0.0):
        return scores

    average = scores / total_num_sentences
    return average


def _preprocess_sentences(
    reference_corpus: Sequence[Union[str, Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
    language: Union[Literal["en"], Literal["ja"]],
) -> Tuple[List[str], List[str]]:
    """Proprocess strings according to language requirements.

    Args:
        reference_corpus: An iterable of iterables of reference corpus.
        hypothesis_corpus: An iterable of hypothesis corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Tuple of lists that contain the cleaned strings for reference_corpus and hypothesis_corpus

    Raises:
        ValueError: If a different language than 'en" or 'ja' is used
        ValueError: If length of reference_corpus not equal to length of hypothesis_corpus
        ValueError: If objects in reference and hypothesis corpus are not strings
    """
    # sanity checks
    reference_corpus, hypothesis_corpus = _validate_inputs(reference_corpus, hypothesis_corpus)

    # preprocess string
    if language == "en":
        preprocess_function = _preprocess_en
    elif language == "ja":
        preprocess_function = _preprocess_ja
    else:
        raise ValueError(f"Expected argument `language` to either be `en` or `ja` but got {language}")

    reference_corpus = [[preprocess_function(ref) for ref in reference] for reference in reference_corpus]
    hypothesis_corpus = [preprocess_function(hyp) for hyp in hypothesis_corpus]

    return reference_corpus, hypothesis_corpus


def _compute_sentence_statistics(
    reference_words: List[List[str]],
    hypothesis: List[str],
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute scores for EED.

    Args:
        reference_words:
            An iterable of reference words
        hypothesis:
            A sentence
        alpha:
            optimal jump penalty, penalty for jumps between characters
        rho:
            coverage cost, penalty for repetition of characters
        deletion:
            penalty for deletion of character
        insertion:
            penalty for insertion or substitution of character

    Returns:
        Tuple of scores and number of sentences as floats
    """
    scores = 0.0
    num_sentences = 0.0

    for reference in reference_words:
        score = _eed_function(reference, hypothesis, alpha, rho, deletion, insertion)
        scores += score
        num_sentences += 1.0

    return scores, num_sentences


def _eed_update(
    reference_corpus: Sequence[Union[str, Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
    language: Literal["en", "ja"] = "en",
    return_sentence_level_score: bool = False,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute scores for EED.

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus
        hypothesis_corpus:
            An iterable of hypothesis corpus
        language:
            Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score:
            An indication of whether sentence-level EED is to be returned
        alpha:
            optimal jump penalty, penalty for jumps between characters
        rho:
            coverage cost, penalty for repetition of characters
        deletion:
            penalty for deletion of character
        insertion:
            penalty for insertion or substitution of character

    Returns:
        Tuple of scores, total sentences as floats, and individual sentence scores
    """
    reference_corpus, hypothesis_corpus = _preprocess_sentences(reference_corpus, hypothesis_corpus, language)

    # check if reference_corpus or hypothesis_corpus is empty
    if 0 in (len(hypothesis_corpus), len(reference_corpus[0])):
        scores = tensor(0.0)
        total_num_sentences = tensor(0.0)
        sentence_eed = []
        return scores, total_num_sentences, sentence_eed

    # calculate score
    scores = 0.0
    total_num_sentences = 0.0

    sentence_eed: Optional[List[Tensor]] = [] if return_sentence_level_score else None 

    for reference_words, hypothesis in zip(reference_corpus, hypothesis_corpus):
        score, num_sentences = _compute_sentence_statistics(
            reference_words, hypothesis, alpha, rho, deletion, insertion
        )

        if sentence_eed is not None:
            sentence_eed.append(_eed_compute(tensor(scores), tensor(total_num_sentences)).unsqueeze(0))

        scores += score
        total_num_sentences += num_sentences

    return tensor(scores), tensor(total_num_sentences), sentence_eed


def eed(
    reference_corpus: Sequence[Union[str, Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
    language: Literal["en", "ja"] = "en",
    return_sentence_level_score: bool = False,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Tensor:
    """Computes extended edit distance score (`EED`_) [1] for strings or list of strings. The metric utilises the
    Levenshtein distance and extends it by adding an additional jump operation.

    Args:
        reference_corpus:
            An iterable of iterables of reference corpus.
        hypothesis_corpus:
            An iterable of hypothesis corpus.
        language:
            Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score:
            An indication of whether sentence-level EED is to be returned.
        alpha:
            optimal jump penalty, penalty for jumps between characters
        rho:
            coverage cost, penalty for repetition of characters
        deletion:
            penalty for deletion of character
        insertion:
            penalty for insertion or substitution of character

    Returns:
        Extended edit distance score as a tensor

    Example:
        >>> from torchmetrics.functional import eed
        >>> reference_corpus = ["this is the reference", "here is another one"]
        >>> hypothesis_corpus = ["this is the prediction", "here is an other sample"]
        >>> eed(reference_corpus=reference_corpus, hypothesis_corpus=hypothesis_corpus)
        tensor(0.3204)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”,
        submitted to WMT 2019. `EED`_
    """
    scores, total_num_sentences, sentence_eed = _eed_update(
        reference_corpus, hypothesis_corpus, language, return_sentence_level_score, alpha, rho, deletion, insertion
    )

    if return_sentence_level_score:
        return sentence_eed

    average = _eed_compute(scores, total_num_sentences)
    return average
