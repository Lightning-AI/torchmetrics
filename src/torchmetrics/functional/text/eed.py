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

from torch import Tensor, stack, tensor
from typing_extensions import Literal

from torchmetrics.functional.text.helper import _validate_inputs


def _distance_between_words(preds_word: str, target_word: str) -> int:
    """Distance measure used for substitutions/identity operation.

    Code adapted from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        preds_word: hypothesis word string
        target_word: reference word string

    Return:
        0 for match, 1 for no match
    """
    return int(preds_word != target_word)


def _eed_function(
    hyp: str,
    ref: str,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> float:
    """Compute extended edit distance score for two lists of strings: hyp and ref.

    Code adapted from: https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        hyp: A hypothesis string
        ref: A reference string
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
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
                    row[i - 1] + _distance_between_words(hyp[i - 1], ref[w - 1]),
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
    """Preprocess english sentences.

    Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

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

    # add space to beginning and end of string
    return " " + sentence + " "


def _preprocess_ja(sentence: str) -> str:
    """Preprocess japanese sentences.

    Copy from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.
    """
    if not isinstance(sentence, str):
        raise ValueError(f"Only strings allowed during preprocessing step, found {type(sentence)} instead")

    sentence = sentence.rstrip()  # trailing space, tab, newline
    # characters which look identical actually are identical
    return unicodedata.normalize("NFKC", sentence)


def _eed_compute(sentence_level_scores: List[Tensor]) -> Tensor:
    """Reduction for extended edit distance.

    Args:
        sentence_level_scores: list of sentence-level scores as floats

    Return:
        average of scores as a tensor
    """
    if len(sentence_level_scores) == 0:
        return tensor(0.0)

    return sum(sentence_level_scores) / tensor(len(sentence_level_scores))


def _preprocess_sentences(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Union[Literal["en"], Literal["ja"]],
) -> Tuple[Union[str, Sequence[str]], Sequence[Union[str, Sequence[str]]]]:
    """Preprocess strings according to language requirements.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Return:
        Tuple of lists that contain the cleaned strings for target and preds

    Raises:
        ValueError: If a different language than ``'en'`` or ``'ja'`` is used
        ValueError: If length of target not equal to length of preds
        ValueError: If objects in reference and hypothesis corpus are not strings
    """
    # sanity checks
    target, preds = _validate_inputs(hypothesis_corpus=preds, ref_corpus=target)

    # preprocess string
    if language == "en":
        preprocess_function = _preprocess_en
    elif language == "ja":
        preprocess_function = _preprocess_ja
    else:
        raise ValueError(f"Expected argument `language` to either be `en` or `ja` but got {language}")

    preds = [preprocess_function(pred) for pred in preds]
    target = [[preprocess_function(ref) for ref in reference] for reference in target]

    return preds, target


def _compute_sentence_statistics(
    preds_word: str,
    target_words: Union[str, Sequence[str]],
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Tensor:
    """Compute scores for ExtendedEditDistance.

    Args:
        target_words: An iterable of reference words
        preds_word: A hypothesis word
        alpha: An optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        best_score: best (lowest) sentence-level score as a Tensor
    """
    best_score = inf

    for reference in target_words:
        score = _eed_function(preds_word, reference, alpha, rho, deletion, insertion)
        if score < best_score:
            best_score = score

    return tensor(best_score)


def _eed_update(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"] = "en",
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
    sentence_eed: Optional[List[Tensor]] = None,
) -> List[Tensor]:
    """Compute scores for ExtendedEditDistance.

    Args:
        preds: An iterable of hypothesis corpus
        target: An iterable of iterables of reference corpus
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character
        sentence_eed: list of sentence-level scores

    Return:
        individual sentence scores as a list of Tensors
    """
    preds, target = _preprocess_sentences(preds, target, language)

    if sentence_eed is None:
        sentence_eed = []

    # return tensor(0.0) if target or preds is empty
    if 0 in (len(preds), len(target[0])):
        return sentence_eed

    for hypothesis, target_words in zip(preds, target):
        score = _compute_sentence_statistics(hypothesis, target_words, alpha, rho, deletion, insertion)
        sentence_eed.append(score)

    return sentence_eed


def extended_edit_distance(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"] = "en",
    return_sentence_level_score: bool = False,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute extended edit distance score (`ExtendedEditDistance`_) [1] for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned.
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as a tensor

    Example:
        >>> from torchmetrics.functional.text import extended_edit_distance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> extended_edit_distance(preds=preds, target=target)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”,
        submitted to WMT 2019. `ExtendedEditDistance`_
    """
    # input validation for parameters
    for param_name, param in zip(["alpha", "rho", "deletion", "insertion"], [alpha, rho, deletion, insertion]):
        if not isinstance(param, float) or isinstance(param, float) and param < 0:
            raise ValueError(f"Parameter `{param_name}` is expected to be a non-negative float.")

    sentence_level_scores = _eed_update(preds, target, language, alpha, rho, deletion, insertion)

    average = _eed_compute(sentence_level_scores)

    if return_sentence_level_score:
        return average, stack(sentence_level_scores)
    return average
