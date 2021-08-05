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

from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures


def wer(
    references: Union[str, List[str]],
    predictions: Union[str, List[str]],
    concatenate_texts: bool = False,
) -> float:
    """Word error rate (WER_) is a common metric of the performance of an automatic speech recognition system. This
    value indicates the percentage of words that were incorrectly predicted. The lower the value, the better the
    performance of the ASR system with a WER of 0 being a perfect score.

    Args:
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts: Whether to concatenate all input texts or compute WER iteratively.

    Returns:
        (float): the word error rate, or if ``return_measures`` is True, we include the incorrect and total.

    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer(predictions=predictions, references=references)
        0.5
    """
    if not _JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "wer metric requires that jiwer is installed."
            " Either install as `pip install torchmetrics[text]` or `pip install jiwer`"
        )
    if concatenate_texts:
        return compute_measures(references, predictions)["wer"]
    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total
