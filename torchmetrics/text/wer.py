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

from typing import List

from torchmetrics.metric import Metric

from torchmetrics.utilities.imports import _module_available

_JIWER_AVAILABLE: bool = _module_available("jiwer")

try:
    from jiwer import compute_measures
except ImportError:
    compute_measures = 0
    Exception("Jiwer not found")


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
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.
    Returns:
        (float): the word error rate
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer = WER(predictions=predictions, references=references)
        >>> wer_score = wer.compute()
        >>> print(wer_score)
        0.5
    """

    def __init__(self, concatenate_texts: bool = False):
        super().__init__()
        self.concatenate_texts = concatenate_texts

    def update(self, preds: List[str], target: List[str]) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> float:
        if self.concatenate_texts:
            return compute_measures(self.target, self.preds)["wer"]
        incorrect = 0
        total = 0
        for prediction, reference in zip(self.preds, self.target):
            measures = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        return incorrect / total
