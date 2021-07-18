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

from torchmetrics.utilities.imports import _module_available

_JIWER_AVAILABLE: bool = _module_available("jiwer")

try:
    from jiwer import compute_measures
except ImportError:
    compute_measures = 0
    Exception("Jiwer not found")


def wer(references: List[str], predictions: List[str], concatenate_texts: bool = False) -> float:
    """
    Args:
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.
    Returns:
        (float): the word error rate
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer_score = wer(predictions=predictions, references=references)
        >>> print(wer_score)
        0.5
    """
    if concatenate_texts:
        return compute_measures(references, predictions)["wer"]
    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total
