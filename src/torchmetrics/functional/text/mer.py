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

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.helper import _edit_distance


def _mer_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:
    """Update the mer score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred.split()
        tgt_tokens = tgt.split()
        errors += _edit_distance(pred_tokens, tgt_tokens)
        total += max(len(tgt_tokens), len(pred_tokens))

    return errors, total


def _mer_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the match error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references

    Returns:
        Match error rate score
    """
    return errors / total


def match_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Match error rate is a metric of the performance of an automatic speech recognition system.

    This value indicates the percentage of words that were incorrectly predicted and inserted. The lower the value, the
    better the performance of the ASR system with a MatchErrorRate of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Match error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> match_error_rate(preds=preds, target=target)
        tensor(0.4444)
    """
    errors, total = _mer_update(
        preds,
        target,
    )
    return _mer_compute(errors, total)
