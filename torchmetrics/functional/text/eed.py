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
from typing import List, Tuple, Union

from torch import tensor


def distance(refWord: str, hypWord: str) -> int:
    """Distance measure used for substitutions/identity operation Copied from
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        refWord: reference word string
        hypWord: hypothesis word string

    Returns:
        0 for match, 1 for no match
    """
    if refWord == hypWord:
        return 0
    return 1


def _eed_function(hyp: str, ref: str) -> float:
    """Computes extended edit distance score for two strings: hypotheses and references. Copied from
    https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        hyp: Transcription to score as a string
        ref: Reference for input as a string

    Returns:
        Extended edit distance score as float
    """
    hyp.insert(0, " ")
    hyp.append(" ")
    ref.insert(0, " ")
    ref.append(" ")

    alpha = 2.0  # optimal jump penalty
    deletion = 0.2

    # substitutions are implemented via the distance function
    insertion = 1.0
    rho = 0.3  # coverage cost
    lj = [-1] * (len(hyp) + 1)

    row = [1] * (len(hyp) + 1)  # row[i] stores cost of cheapest path from (0,0) to (i,l) in CDER aligment grid.

    row[0] = 0  # CDER initialisation 0,0 = 0 rest 1
    nextRow = [inf] * (len(hyp) + 1)

    for w in range(1, len(ref) + 1):
        for i in range(0, len(hyp) + 1):

            if i > 0:
                nextRow[i] = min(
                    nextRow[i - 1] + deletion,
                    row[i - 1] + distance(ref[w - 1], hyp[i - 1]),
                    row[i] + insertion,
                )
            else:
                nextRow[i] = row[i] + 1.0

        minInd = nextRow.index(min(nextRow))
        lj[minInd] += 1

        # Long Jumps
        if ref[w - 1] == " ":
            jump = alpha + nextRow[minInd]
            nextRow = [x if x < jump else jump for x in nextRow]

        row = nextRow
        nextRow = [inf] * (len(hyp) + 1)

    coverage = rho * sum(x if x >= 0 else 1 for x in lj)

    return min(1, (row[-1] + coverage) / (float(len(ref)) + coverage))


def preprocess_en(s: str) -> str:
    """Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py."""
    if isinstance(s, str) is not True:
        raise RuntimeError(f"Only strings allowed during preprocessing step, found {type(s)} instead")

    s = s.rstrip()  # trailing space, tab, or newline

    s = s.replace(".", " .")
    s = s.replace("!", " !")
    s = s.replace("?", " ?")
    s = s.replace(",", " ,")

    s = re.sub(r"\s+", r" ", s)  # get rid of extra spaces
    s = re.sub(r"(\d) ([.,]) (\d)", r"\1\2\3", s)  # 0 . 1 -> 0.1
    s = re.sub(r"(Dr|Jr|Prof|Rev|Gen|Mr|Mt|Mrs|Ms) .", r"\1.", s)  # Mr . -> Mr.
    s = s.replace("e . g .", "e.g.")
    s = s.replace("i . e .", "e.g.")
    s = s.replace("U . S .", "U.S.")
    return s


def preprocess_ja(s: str) -> str:
    """Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py."""
    if isinstance(s, str) is not True:
        raise RuntimeError(f"Only strings allowed during preprocessing step, found {type(s)} instead")

    s = s.rstrip()  # trailing space, tab, newline
    s = unicodedata.normalize("NFKC", s)  # まず正規化
    return s


def _eed_compute(scores: tensor, total: tensor) -> tensor:
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
    language: str,
) -> Tuple[List[str], List[str]]:
    """Proprocess strings according to language requirements.

    Args:
        hypotheses: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each input as a string or list of strings
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Tuple of lists that contain the cleaned strings for hypotheses and references
    """
    # sanity checks
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    if isinstance(references, str):
        references = [references]

    if len(hypotheses) != len(references):
        raise RuntimeError("Length of hypotheses must equal length of references")

    # preprocess string
    if language == "en":
        preprocess_function = preprocess_en
    elif language == "ja":
        preprocess_function = preprocess_ja
    else:
        raise RuntimeError(f"Language {language} not supported, supported languages are 'en' and 'ja'")

    hypotheses = [preprocess_function(string) for string in hypotheses]
    references = [preprocess_function(string) for string in references]

    return hypotheses, references


def _eed_update(
    hypotheses: Union[str, List[str]],
    references: Union[str, List[str]],
    language: str,
) -> Tuple[tensor, tensor]:
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

    for hyp, ref in zip(hypotheses, references):
        hyp, ref = list(hyp), list(ref)
        score = _eed_function(hyp, ref)
        scores += score
        total += 1.0

    return tensor(scores), tensor(total)


def eed(
    hypotheses: Union[str, List[str]],
    references: Union[str, List[str]],
    language: str = "en",
) -> tensor:
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
