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

from typing import List, Tuple, Union

from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures


def wer(
    references: Union[str, List[str]],
    predictions: Union[str, List[str]],
    concatenate_texts: bool = False,
    return_measures: bool = False
) -> Union[float, Tuple[float, int, int]]:
    """
    Args:
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.
        return_measures (bool, default=False): Return the number of incorrect and total in WER calculation.
    Returns:
        (float): the word error rate, or if ``return_measures`` is True, we include the incorrect and total.
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer(predictions=predictions, references=references)
        0.5
        >>> wer(predictions=predictions, references=references, return_measures=True)
        (0.5, 4, 8)

    """
    if concatenate_texts:
        return compute_measures(references, predictions)["wer"]
    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    if return_measures:
        return (incorrect / total), incorrect, total
    return incorrect / total
