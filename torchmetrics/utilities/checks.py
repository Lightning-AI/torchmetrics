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
from typing import Tuple

import torch

from torchmetrics.utilities.data import to_onehot


def _check_same_shape(pred: torch.Tensor, target: torch.Tensor):
    """ Check that predictions and target have the same shape, else raise error """
    if pred.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")


def _input_format_classification(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert preds and target tensors into label tensors

    Args:
        preds: either tensor with labels, tensor with probabilities/logits or multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input

    Returns:
        preds: tensor with labels
        target: tensor with labels

    Example:
        >>> _input_format_classification(torch.tensor([[0.45, 0.55], [0.3, 0.7], [0.9, 0.1]]), torch.tensor([1, 0, 0]))
        (tensor([1, 1, 0]), tensor([1, 0, 0]))
    """
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds and target must have same number of dimensions, or one additional dimension for preds")

    if preds.ndim == target.ndim + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        # binary or multilabel probablities
        preds = (preds >= threshold).long()
    return preds, target


def _input_format_classification_one_hot(
    num_classes: int,
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    multilabel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert preds and target tensors into one hot spare label tensors

    Args:
        num_classes: number of classes
        preds: either tensor with labels, tensor with probabilities/logits or multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input
        multilabel: boolean flag indicating if input is multilabel

    Returns:
        preds: one hot tensor of shape [num_classes, -1] with predicted labels
        target: one hot tensors of shape [num_classes, -1] with true labels
    """
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds and target must have same number of dimensions, or one additional dimension for preds")

    if preds.ndim == target.ndim + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if preds.ndim == target.ndim and preds.dtype in (torch.long, torch.int) and num_classes > 1 and not multilabel:
        # multi-class
        preds = to_onehot(preds, num_classes=num_classes)
        target = to_onehot(target, num_classes=num_classes)

    elif preds.ndim == target.ndim and preds.is_floating_point():
        # binary or multilabel probablities
        preds = (preds >= threshold).long()

    # transpose class as first dim and reshape
    if preds.ndim > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)

    return preds.reshape(num_classes, -1), target.reshape(num_classes, -1)
