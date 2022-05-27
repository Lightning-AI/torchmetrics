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

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _bincount
from torchmetrics.utilities.prints import rank_zero_warn


def _confusion_matrix_reduce(confmat: Tensor, normalize: Optional[str] = None, multilabel: bool = False) -> Tensor:
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
            confmat = confmat / confmat.sum(axis=[1, 2] if multilabel else [0, 1])

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
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = bool
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
            "Detected the following values in `target`: {unique_values} but expected only"
            " the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                "Detected the following values in `preds`: {unique_values} but expected only"
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
        if not ((0 <= preds) * (preds <= 1)).all():
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
        _binary_confusion_matrix_tensor_validation(preds, target, threshold, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _binary_confusion_matrix_compute(confmat, normalize)
