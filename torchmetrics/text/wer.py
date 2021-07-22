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

from typing import List, Union

from torchmetrics.functional import wer
from torchmetrics.metric import Metric


class WER(Metric):
    r"""
    `Word error rate (WER) <https://en.wikipedia.org/wiki/Word_error_rate>`_ is a common metric of
    the performance of an automatic speech recognition system.
    This value indicates the percentage of words that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.
    Word error rate can then be computed as:

    .. math::
        WER = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where:

        - S is the number of substitutions,
        - D is the number of deletions,
        - I is the number of insertions,
        - C is the number of correct words,
        - N is the number of words in the reference (N=S+D+C).

    Compute WER score of transcribed segments against references.

    Args:
        concatenate_texts: Whether to concatenate all input texts or compute WER iteratively.
    Returns:
        (float): the word error rate

    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> metric = WER()
        >>> metric(predictions, references)
        0.5

    """

    def __init__(self, concatenate_texts: bool = False):
        super().__init__()
        self.concatenate_texts = concatenate_texts
        self.add_state('predictions', [])
        self.add_state('references', [])

    def update(self, predictions: Union[str, List[str]], references: Union[str, List[str]]) -> None:
        """
        Store predictions/references for computing Word Error Rate scores.
        Args:
            predictions: List of transcriptions to score.
            references: List of references for each speech input.
        """
        self.predictions.append(predictions)
        self.references.append(references)

    def compute(self) -> float:
        """
        Calculate Word Error Rate scores.

        Return:
            Float with WER Score.
        """
        if self.concatenate_texts:
            return wer(self.references, self.predictions)
        incorrect = 0
        total = 0
        for prediction, reference in zip(self.predictions, self.references):
            _, pred_incorrect, pred_total = wer(reference, prediction, return_measures=True)
            incorrect += pred_incorrect
            total += pred_total
        return incorrect / total
