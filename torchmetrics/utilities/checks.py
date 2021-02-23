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