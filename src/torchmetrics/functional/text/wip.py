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
from typing import List, Tuple, Union

from torch import Tensor, tensor

from torchmetrics.functional.text.helper import _edit_distance


def _wip_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update the wip score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
        Number of words overall prediction
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    total = tensor(0.0)
    errors = tensor(0.0)
    target_total = tensor(0.0)
    preds_total = tensor(0.0)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred.split()
        target_tokens = tgt.split()
        errors += _edit_distance(pred_tokens, target_tokens)
        target_total += len(target_tokens)
        preds_total += len(pred_tokens)
        total += max(len(target_tokens), len(pred_tokens))

    return errors - total, target_total, preds_total


def _wip_compute(errors: Tensor, target_total: Tensor, preds_total: Tensor) -> Tensor:
    """Compute the Word Information Perserved.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        target_total: Number of words overall references
        preds_total: Number of words overall prediction

    Returns:
        Word Information Perserved score
    """
    return (errors / target_total) * (errors / preds_total)


def word_information_preserved(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Word Information Preserved rate is a metric of the performance of an automatic speech recognition system.

    This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the
    better the performance of the ASR system with a Word Information preserved rate of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Word Information preserved rate

    Examples:
        >>> from torchmetrics.functional.text import word_information_preserved
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_information_preserved(preds, target)
        tensor(0.3472)
    """
    errors, reference_total, prediction_total = _wip_update(preds, target)
    return _wip_compute(errors, reference_total, prediction_total)
