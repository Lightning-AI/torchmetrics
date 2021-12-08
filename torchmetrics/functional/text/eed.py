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
from typing import List, Tuple, Union, Literal

from torch import tensor, Tensor


def _distance_between_words(reference_word: str, hypothesis_word: str) -> int:
    """Distance measure used for substitutions/identity operation Copied from
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        reference_word: reference word string
        hypothesis_word: hypothesis word string

    Returns:
        0 for match, 1 for no match
    """
    return int(reference_word != hypothesis_word)


def _eed_function(
    hyp: List[str],
    ref: List[str],
    alpha: float = 2.0,  # optimal jump penalty
    deletion: float = 0.2,
    insertion: float = 1.0,
    rho: float = 0.3,  # coverage cost
) -> float:
    """Computes extended edit distance score for two strings: hypotheses and references. Copied from
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        hyp: Transcription to score as a string
        ref: Reference for input as a string

    Returns:
        Extended edit distance score as float
    """
    lj = [-1] * (len(hyp) + 1)

    # row[i] stores cost of cheapest path from (0,0) to (i,l) in CDER aligment grid.
    row = [1.0] * (len(hyp) + 1)

    row[0] = 0.0  # CDER initialisation 0,0 = 0, rest 1
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

        minInd = next_row.index(min(next_row))
        lj[minInd] += 1

        # Long Jumps
        if ref[w - 1] == " ":
            jump = alpha + next_row[minInd]
            next_row = [min(x, jump) for x in next_row]

        row = next_row
        next_row = [inf] * (len(hyp) + 1)

    coverage = rho * sum(x if x >= 0 else 1 for x in lj)

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


def _eed_compute(scores: Tensor, total: Tensor) -> Tensor:
    """Final step in extended edit distance.

    Args:
        scores: sum of individual sentence scores as a tensor
        total: number of sentences as a tensor

    Returns:
        average of scores as a tensor
    """
    average = scores / total
    return average


def _preprocess_sentences(
    hypotheses: Union[str, List[str]],
    references: Union[str, List[str]],
    language: Union[Literal["en"], Literal["ja"]],
) -> Tuple[List[str], List[str]]:
    """Proprocess strings according to language requirements.

    Args:
        hypotheses: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each input as a string or list of strings
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Tuple of lists that contain the cleaned strings for hypotheses and references

    Raises:
        ValueError: If a different language than 'en" or 'ja' is used
        ValueError: If lenght of hypotheses is not equal to length of references
    """
    # sanity checks
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    if isinstance(references, str):
        references = [references]

    if len(hypotheses) != len(references):
        raise ValueError("Length of hypotheses must equal length of references")

    # preprocess string
    if language == "en":
        preprocess_function = _preprocess_en
    elif language == "ja":
        preprocess_function = _preprocess_ja
    else:
        raise ValueError(f"Language {language} not supported, supported languages are 'en' and 'ja'")

    hypotheses = [preprocess_function(hyp) for hyp in hypotheses]
    references = [preprocess_function(ref) for ref in references]

    return hypotheses, references


def _eed_update(
    hypotheses: Union[str, List[str]],
    references: Union[str, List[str]],
    language: Literal["en", "ja"] = "en",
) -> Tuple[Tensor, Tensor]:
    """Compute scores for EED.

    Args:
        hypotheses: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each input as a string or list of strings
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Tuple of scores and total sentences as floats
    """
    hypotheses, references = _preprocess_sentences(hypotheses, references, language)

    # calculate score
    scores = 0.0
    total = 0.0

    for hypothesis, reference in zip(hypotheses, references):
        hyp: List[str] = list(hypothesis)
        ref: List[str] = list(reference)
        score = _eed_function(hyp, ref)
        scores += score
        total += 1.0

    return tensor(scores), tensor(total)


def eed(
    hypotheses: Union[str, List[str]],
    references: Union[str, List[str]],
    language: Literal["en", "ja"] = "en",
) -> Tensor:
    """Computes extended edit distance score (`EED`_) [1] for strings or list of strings The metric utilises the
    Levenshtein distance and extends it by adding an additional jump operation.

    Args:
        hypotheses: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each input as a string or list of strings
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Extended edit distance score as a tensor

    Example:
        >>> hypotheses = ["this is the prediction", "here is an other sample"]
        >>> references = ["this is the reference", "here is another one"]
        >>> eed(hypotheses=hypotheses, references=references)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”, submitted
        to WMT 2019. `EED`_
    """
    scores, total = _eed_update(hypotheses, references, language)
    average = _eed_compute(scores, total)

    return average
