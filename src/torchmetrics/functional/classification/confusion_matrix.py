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
from typing import Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn


def _confusion_matrix_reduce(
    confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None
) -> Tensor:
    """Reduce an un-normalized confusion matrix.

    Args:
        confmat: un-normalized confusion matrix
        normalize: normalization method.
            - `"true"` will divide by the sum of the column dimension.
            - `"pred"` will divide by the sum of the row dimension.
            - `"all"` will divide by the sum of the full matrix
            - `"none"` or `None` will apply no reduction.

    Returns:
        Normalized confusion matrix
    """
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Argument `normalize` needs to one of the following: {allowed_normalize}")
    if normalize is not None and normalize != "none":
        confmat = confmat.float() if not confmat.is_floating_point() else confmat
        if normalize == "true":
            confmat = confmat / confmat.sum(dim=-1, keepdim=True)
        elif normalize == "pred":
            confmat = confmat / confmat.sum(dim=-2, keepdim=True)
        elif normalize == "all":
            confmat = confmat / confmat.sum(dim=[-2, -1], keepdim=True)

        nan_elements = confmat[torch.isnan(confmat)].nelement()
        if nan_elements:
            confmat[torch.isnan(confmat)] = 0
            rank_zero_warn(f"{nan_elements} NaN values found in confusion matrix have been replaced with zeros.")
    return confmat


def _binary_confusion_matrix_arg_validation(
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``ignore_index`` has to be None or int
    - ``normalize`` has to be "true" | "pred" | "all" | "none" | None
    """
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _binary_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    """
    # Check that they have same shape
    _check_same_shape(preds, target)

    # Check that target only contains {0,1} values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains {0,1} values
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
    convert_to_labels: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - Remove all datapoints that should be ignored
    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
        if convert_to_labels:
            preds = preds > threshold

    return preds, target


def _binary_confusion_matrix_update(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (target * 2 + preds).to(torch.long)
    bins = _bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)


def _binary_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None
) -> Tensor:
    """Reduces the confusion matrix to it's final form.

    Normalization technique can be chosen by ``normalize``.
    """
    return _confusion_matrix_reduce(confmat, normalize)


def binary_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `confusion matrix`_ for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A ``[2, 2]`` tensor

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_confusion_matrix
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> binary_confusion_matrix(preds, target)
        tensor([[2, 0],
                [1, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_confusion_matrix
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_confusion_matrix(preds, target)
        tensor([[2, 0],
                [1, 1]])
    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _binary_confusion_matrix_compute(confmat, normalize)


def _multiclass_confusion_matrix_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``ignore_index`` has to be None or int
    - ``normalize`` has to be "true" | "pred" | "all" | "none" | None
    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _multiclass_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - if target has one more dimension than preds, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}
    """
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

    num_unique_values = len(torch.unique(target))
    check = num_unique_values > num_classes if ignore_index is None else num_unique_values > num_classes + 1
    if check:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes if ignore_index is None else num_classes + 1} but found "
            f"{num_unique_values} in `target`."
        )

    if not preds.is_floating_point():
        num_unique_values = len(torch.unique(preds))
        if num_unique_values > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected only "
                f"{num_classes} but found {num_unique_values} in `preds`."
            )


def _multiclass_confusion_matrix_format(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
    convert_to_labels: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - Applies argmax if preds have one more dimension than target
    - Remove all datapoints that should be ignored
    """
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and convert_to_labels:
        preds = preds.argmax(dim=1)

    preds = preds.flatten() if convert_to_labels else torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
    target = target.flatten()

    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    return preds, target


def _multiclass_confusion_matrix_update(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = target.to(torch.long) * num_classes + preds.to(torch.long)
    bins = _bincount(unique_mapping, minlength=num_classes**2)
    return bins.reshape(num_classes, num_classes)


def _multiclass_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None
) -> Tensor:
    """Reduces the confusion matrix to it's final form.

    Normalization technique can be chosen by ``normalize``.
    """
    return _confusion_matrix_reduce(confmat, normalize)


def multiclass_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `confusion matrix`_ for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A ``[num_classes, num_classes]`` tensor

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_confusion_matrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_confusion_matrix(preds, target, num_classes=3)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_confusion_matrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_confusion_matrix(preds, target, num_classes=3)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])
    """
    if validate_args:
        _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _multiclass_confusion_matrix_compute(confmat, normalize)


def _multilabel_confusion_matrix_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` should be an int larger than 1
    - ``threshold`` has to be a float in the [0,1] range
    - ``ignore_index`` has to be None or int
    - ``normalize`` has to be "true" | "pred" | "all" | "none" | None
    """
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}")
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.")


def _multilabel_confusion_matrix_tensor_validation(
    preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - the second dimension of both tensors need to be equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    """
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
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
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
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    should_threshold: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all elements that should be ignored with negative numbers for later filtration
    """
    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()
        if should_threshold:
            preds = preds > threshold
    preds = torch.movedim(preds, 1, -1).reshape(-1, num_labels)
    target = torch.movedim(target, 1, -1).reshape(-1, num_labels)

    if ignore_index is not None:
        preds = preds.clone()
        target = target.clone()
        # Make sure that when we map, it will always result in a negative number that we can filter away
        # Each label correspond to a 2x2 matrix = 4 elements per label
        idx = target == ignore_index
        preds[idx] = -4 * num_labels
        target[idx] = -4 * num_labels

    return preds, target


def _multilabel_confusion_matrix_update(preds: Tensor, target: Tensor, num_labels: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = ((2 * target + preds) + 4 * torch.arange(num_labels, device=preds.device)).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels)
    return bins.reshape(num_labels, 2, 2)


def _multilabel_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None
) -> Tensor:
    """Reduces the confusion matrix to it's final form.

    Normalization technique can be chosen by ``normalize``.
    """
    return _confusion_matrix_reduce(confmat, normalize)


def multilabel_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `confusion matrix`_ for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A ``[num_labels, 2, 2]`` tensor

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_confusion_matrix
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_confusion_matrix(preds, target, num_labels=3)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_confusion_matrix
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_confusion_matrix(preds, target, num_labels=3)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])
    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index, normalize)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _multilabel_confusion_matrix_compute(confmat, normalize)


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `confusion matrix`_.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`binary_confusion_matrix`, :func:`multiclass_confusion_matrix` and :func:`multilabel_confusion_matrix` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import ConfusionMatrix
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(task="binary")
        >>> confmat(preds, target)
        tensor([[2, 0],
                [1, 1]])

        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> confmat = ConfusionMatrix(task="multiclass", num_classes=3)
        >>> confmat(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> confmat = ConfusionMatrix(task="multilabel", num_labels=3)
        >>> confmat(preds, target)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])
    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_confusion_matrix(preds, target, threshold, normalize, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_confusion_matrix(preds, target, num_classes, normalize, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_confusion_matrix(preds, target, num_labels, threshold, normalize, ignore_index, validate_args)
    raise ValueError(f"Task {task} not supported.")
