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
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.prints import rank_zero_warn


def _confusion_matrix_reduce(confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None, multilabel: bool = False) -> Tensor:
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Argument `normalize` needs to one of the following: {allowed_normalize}")
    if normalize is not None and normalize != "none":
        confmat = confmat.float() if not confmat.is_floating_point() else confmat
        if normalize == "true":
            confmat = confmat / confmat.sum(axis=2 if multilabel else 1, keepdim=True)
        elif normalize == "pred":
            confmat = confmat / confmat.sum(axis=1 if multilabel else 0, keepdim=True)
        elif normalize == "all":
            confmat = confmat / confmat.sum(axis=[1, 2] if multilabel else [0, 1], keepdim=True)

        nan_elements = confmat[torch.isnan(confmat)].nelement()
        if nan_elements != 0:
            confmat[torch.isnan(confmat)] = 0
            rank_zero_warn(f"{nan_elements} NaN values found in confusion matrix have been replaced with zeros.")
    return confmat


def _binary_confusion_matrix_arg_validation(
    threshold: float = 0.5, ignore_index: Optional[int] = None, normalize: Optional[str] = None
) -> None:
    """Validate non tensor input."""
    if not isinstance(threshold, float):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _binary_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input."""
    # Check that they have same shape
    _check_same_shape(preds, target)

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )


def _binary_confusion_matrix_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if preds.is_floating_point():
        if not torch.all((0 <= preds) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
        preds = preds > threshold

    return preds, target


def _binary_confusion_matrix_update(preds: Tensor, target: Tensor) -> Tensor:
    """Calculate confusion matrix on current input."""
    unique_mapping = (target * 2 + preds).to(torch.long)
    bins = _bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)


def _binary_confusion_matrix_compute(confmat: Tensor, normalize: Optional[str] = None) -> Tensor:
    """Calculate final confusion matrix."""
    return _confusion_matrix_reduce(confmat, normalize, multilabel=False)


def binary_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _binary_confusion_matrix_compute(confmat, normalize)


def _multiclass_confusion_matrix_arg_validation(
    num_classes: int, ignore_index: Optional[int] = None, normalize: Optional[str] = None
) -> None:
    if not isinstance(num_classes, int) and num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _multiclass_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input."""
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds.shape[1]` should be"
                " equal to number of classes."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )
    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    unique_values = torch.unique(target)
    if ignore_index is None:
        check = len(unique_values) > num_classes
    else:
        check = len(unique_values) > num_classes + 1
    if check:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes if ignore_index is None else num_classes + 1} but found"
            f"{len(unique_values)} in `target`."
        )

    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if len(unique_values) > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected only "
                f"{num_classes} but found {len(unique_values)} in `preds`."
            )


def _multiclass_confusion_matrix_format(
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1:
        preds = preds.argmax(dim=1)

    preds = preds.flatten()
    target = target.flatten()

    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    return preds, target


def _multiclass_confusion_matrix_update(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    unique_mapping = (target * num_classes + preds).to(torch.long)
    bins = _bincount(unique_mapping, minlength=num_classes**2)
    return bins.reshape(num_classes, num_classes)


def _multiclass_confusion_matrix_compute(confmat: Tensor, normalize: Optional[str] = None) -> Tensor:
    return _confusion_matrix_reduce(confmat, normalize, multilabel=False)


def multiclass_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _multiclass_confusion_matrix_compute(confmat, normalize)


def _multilabel_confusion_matrix_arg_validation(
    num_labels: int, threshold: float = 0.5, ignore_index: Optional[int] = None, normalize: Optional[str] = None
) -> None:
    if not isinstance(num_labels, int) and num_labels < 2:
        raise ValueError(f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}")
    if not isinstance(threshold, float):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _multilabel_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input."""
    # Check that they have same shape
    _check_same_shape(preds, target)

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )


def _multilabel_confusion_matrix_format(
    preds: Tensor, target: Tensor, num_labels: int, threshold: float = 0.5, ignore_index: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    if preds.is_floating_point():
        if not torch.all((0 <= preds) * (preds <= 1)):
            preds = preds.sigmoid()
        preds = preds > threshold
    preds = preds.movedim(1, -1).reshape(-1, num_labels)
    target = target.movedim(1, -1).reshape(-1, num_labels)

    if ignore_index is not None:
        preds = preds.clone()
        target = target.clone()
        # make sure that when we map, it will always result in a negative number that we can filter away
        idx = target == ignore_index
        preds[idx] = -4 * num_labels
        target[idx] = -4 * num_labels

    return preds, target


def _multilabel_confusion_matrix_update(preds: Tensor, target: Tensor, num_labels: int) -> Tensor:
    unique_mapping = ((2 * target + preds) + 4 * torch.arange(num_labels, device=preds.device)).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels)
    return bins.reshape(num_labels, 2, 2)


def _multilabel_confusion_matrix_compute(confmat: Tensor, normalize: Optional[str] = None) -> Tensor:
    return _confusion_matrix_reduce(confmat, normalize, multilabel=True)


def multilabel_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index, normalize)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _multilabel_confusion_matrix_compute(confmat, normalize)


# -------------------------- Old stuff --------------------------


def _confusion_matrix_update(
    preds: Tensor, target: Tensor, num_classes: int, threshold: float = 0.5, multilabel: bool = False
) -> Tensor:
    """Updates and returns confusion matrix (without any normalization) based on the mode of the input.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the
            case of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        multilabel: determines if data is multilabel or not.
    """

    preds, target, mode = _input_format_classification(preds, target, threshold)
    if mode not in (DataType.BINARY, DataType.MULTILABEL):
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
    if multilabel:
        unique_mapping = ((2 * target + preds) + 4 * torch.arange(num_classes, device=preds.device)).flatten()
        minlength = 4 * num_classes
    else:
        unique_mapping = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
        minlength = num_classes**2

    bins = _bincount(unique_mapping, minlength=minlength)
    if multilabel:
        confmat = bins.reshape(num_classes, 2, 2)
    else:
        confmat = bins.reshape(num_classes, num_classes)
    return confmat


def _confusion_matrix_compute(confmat: Tensor, normalize: Optional[str] = None) -> Tensor:
    """Computes confusion matrix based on the normalization mode.

    Args:
        confmat: Confusion matrix without normalization
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

    Example:
        >>> # binary case
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = _confusion_matrix_update(preds, target, num_classes=2)
        >>> _confusion_matrix_compute(confmat)
        tensor([[2, 0],
                [1, 1]])

        >>> # multiclass case
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> confmat = _confusion_matrix_update(preds, target, num_classes=3)
        >>> _confusion_matrix_compute(confmat)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

        >>> # multilabel case
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> confmat = _confusion_matrix_update(preds, target, num_classes=3, multilabel=True)
        >>> _confusion_matrix_compute(confmat)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])
    """

    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Argument average needs to one of the following: {allowed_normalize}")
    if normalize is not None and normalize != "none":
        confmat = confmat.float() if not confmat.is_floating_point() else confmat
        if normalize == "true":
            confmat = confmat / confmat.sum(axis=1, keepdim=True)
        elif normalize == "pred":
            confmat = confmat / confmat.sum(axis=0, keepdim=True)
        elif normalize == "all":
            confmat = confmat / confmat.sum()

        nan_elements = confmat[torch.isnan(confmat)].nelement()
        if nan_elements != 0:
            confmat[torch.isnan(confmat)] = 0
            rank_zero_warn(f"{nan_elements} nan values found in confusion matrix have been replaced with zeros.")
    return confmat


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    normalize: Optional[str] = None,
    threshold: float = 0.5,
    multilabel: bool = False,
) -> Tensor:
    r"""
    Computes the `confusion matrix`_.  Works with binary,
    multiclass, and multilabel data.  Accepts probabilities or logits from a model output or integer class
    values in prediction. Works with multi-dimensional preds and target, but it should be noted that
    additional dimensions will be flattened.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    If working with multilabel data, setting the ``is_multilabel`` argument to ``True`` will make sure that a
    `confusion matrix gets calculated per label`_.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/logits/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        multilabel:
            determines if data is multilabel or not.

    Example (binary data):
        >>> from torchmetrics import ConfusionMatrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(num_classes=2)
        >>> confmat(preds, target)
        tensor([[2, 0],
                [1, 1]])

    Example (multiclass data):
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> confmat = ConfusionMatrix(num_classes=3)
        >>> confmat(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    Example (multilabel data):
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> confmat = ConfusionMatrix(num_classes=3, multilabel=True)
        >>> confmat(preds, target)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])

    """
    rank_zero_warn(
        "`torchmetrics.functional.confusion_matrix` have been deprecated in v0.10 in favor of"
        "`torchmetrics.functional.binary_confusion_matrix`, `torchmetrics.functional.multiclass_confusion_matrix`"
        "and `torchmetrics.functional.multilabel_confusion_matrix`. Please upgrade to the version that matches"
        "your problem (API may have changed). This function will be removed v0.11.",
        DeprecationWarning,
    )
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold, multilabel)
    return _confusion_matrix_compute(confmat, normalize)
