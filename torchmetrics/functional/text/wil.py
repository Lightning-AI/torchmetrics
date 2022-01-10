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

from torchmetrics.functional.text.helper import _edit_distance


def _wil_update(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update the wil score with the current set of references and predictions.

    Args:
        predictions: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each speech input as a string or list of strings
    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
        Number of words overall predictions
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    total = tensor(0, dtype=torch.float)
    errors = tensor(0, dtype=torch.float)
    reference_total = tensor(0, dtype=torch.float)
    prediction_total = tensor(0, dtype=torch.float)
    for prediction, reference in zip(predictions, references):
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        errors += _edit_distance(prediction_tokens, reference_tokens)
        reference_total += len(reference_tokens)
        prediction_total += len(prediction_tokens)
        total += max(len(reference_tokens), len(prediction_tokens))

    return errors - total, reference_total, prediction_total


def _wil_compute(errors: Tensor, reference_total: Tensor, prediction_total: Tensor) -> Tensor:
    """Compute the Word Information Lost.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        reference_total: Number of words overall references
        prediction_total: Number of words overall prediction
    Returns:
        Word Information Lost score
    """
    return 1 - ((errors / reference_total) * (errors / prediction_total))


def word_information_lost(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> Tensor:
    """Word Information Lost rate is a metric of the performance of an automatic speech recognition system. This
    value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better the
    performance of the ASR system with a Word Information Lost rate of 0 being a perfect score.
    Args:
        predictions: Transcription(s) to score as a string or list of strings
        references: Reference(s) for each speech input as a string or list of strings
    Returns:
        Word Information Lost rate
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> word_information_lost(predictions=predictions, references=references)
        tensor(0.6528)
    """
    errors, reference_total, prediction_total = _wil_update(predictions, references)
    return _wil_compute(errors, reference_total, prediction_total)
