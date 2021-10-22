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

import torch
from torch import Tensor, tensor


def _edit_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    """Standard dynamic programming algorithm to compute the edit distance.

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        (int) Edit distance between the predicted sentence and the reference sentence
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def _cer_update(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:
    """Update the cer score with the current set of references and predictions.

    Args:
        predictions: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each speech input as a string or list of strings
    Returns:
        (Tensor) Number of edit operations to get from the reference to the prediction, summed over all samples
        (Tensor) Number of character over all references
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for prediction, reference in zip(predictions, references):
        prediction_tokens = prediction
        reference_tokens = reference
        errors += _edit_distance(list(prediction_tokens), list(reference_tokens))
        total += len(reference_tokens)
    return errors, total


def _cer_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the Character error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of characters over all references
    Returns:
        (Tensor) Character error rate
    """
    return errors / total


def char_error_rate(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> Tensor:
    """character error rate is a common metric of the performance of an automatic speech recognition system. This
    value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better the
    performance of the ASR system with a CER of 0 being a perfect score.
    Args:
        predictions: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each speech input as a string or list of strings
    Returns:
        (Tensor) Character error rate
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> char_error_rate(predictions=predictions, references=references)
        tensor(0.3415)
    """
    errors, total = _cer_update(predictions, references)
    return _cer_compute(errors, total)
