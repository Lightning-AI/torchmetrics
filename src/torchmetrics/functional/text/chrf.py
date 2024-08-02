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
from typing import Sequence, Tuple, Union

from torch import Tensor


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
    `chrF++ score`_. This implementation follows the implementations from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    .. attention::
        ChrF has been temporarily removed from the TorchMetrics package
        due to licensing issues with the upstream package.

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
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.
        return_sentence_level_score: An indication whether a sentence-level chrF/chrF++ score to be returned.

    References:
        [1] chrF: character n-gram F-score for automatic MT evaluation by Maja Popović `chrF score`_

        [2] chrF++: words helping character n-grams by Maja Popović `chrF++ score`_

    """
    raise NotImplementedError(
        "ChrF has been temporarily removed from the TorchMetrics package"
        " due to licensing issues with the upstream package."
    )
