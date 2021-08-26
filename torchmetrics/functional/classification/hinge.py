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
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import DataType, EnumStr


class MulticlassMode(EnumStr):
    """Enum to represent possible multiclass modes of hinge.

    >>> "Crammer-Singer" in list(MulticlassMode)
    True
    """

    CRAMMER_SINGER = "crammer-singer"
    ONE_VS_ALL = "one-vs-all"


def _check_shape_and_type_consistency_hinge(
    preds: Tensor,
    target: Tensor,
) -> DataType:
    """Checks shape and type of `preds` and `target` and returns mode of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    Raises:
        `ValueError`: if `target` is not one dimensional
        `ValueError`: if `preds` and `target` do not have the same shape in the first dimension
        `ValueError`: if `pred` is neither one nor two dimensional
    """

    if target.ndim > 1:
        raise ValueError(
            f"The `target` should be one dimensional, got `target` with shape={target.shape}.",
        )

    if preds.ndim == 1:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.BINARY
    elif preds.ndim == 2:
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                "The `preds` and `target` should have the same shape in the first dimension,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.MULTICLASS
    else:
        raise ValueError(f"The `preds` should be one or two dimensional, got `preds` with shape={preds.shape}.")
    return mode


def _hinge_update(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    multiclass_mode: Optional[Union[str, MulticlassMode]] = None,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns sum over Hinge loss scores for each observation and the total number of observations.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        squared: If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.
    """
    preds, target = _input_squeeze(preds, target)

    mode = _check_shape_and_type_consistency_hinge(preds, target)

    if mode == DataType.MULTICLASS:
        target = to_onehot(target, max(2, preds.shape[1])).bool()

    if mode == DataType.MULTICLASS and (multiclass_mode is None or multiclass_mode == MulticlassMode.CRAMMER_SINGER):
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    elif mode == DataType.BINARY or multiclass_mode == MulticlassMode.ONE_VS_ALL:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]
    else:
        raise ValueError(
            "The `multiclass_mode` should be either None / 'crammer-singer' / MulticlassMode.CRAMMER_SINGER"
            "(default) or 'one-vs-all' / MulticlassMode.ONE_VS_ALL,"
            f" got {multiclass_mode}."
        )

    measures = 1 - margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def _hinge_compute(measure: Tensor, total: Tensor) -> Tensor:
    """Computes mean Hinge loss.

    Args:
        measure: Sum over hinge losses for each each observation
        total: Number of observations

    Example:
        >>> # binary case
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> measure, total = _hinge_update(preds, target)
        >>> _hinge_compute(measure, total)
        tensor(0.3000)

        >>> # multiclass case
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> measure, total = _hinge_update(preds, target)
        >>> _hinge_compute(measure, total)
        tensor(2.9000)

        >>> # multiclass one-vs-all mode case
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> measure, total = _hinge_update(preds, target, multiclass_mode="one-vs-all")
        >>> _hinge_compute(measure, total)
        tensor([2.2333, 1.5000, 1.2333])
    """

    return measure / total


def hinge(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    multiclass_mode: Optional[Union[str, MulticlassMode]] = None,
) -> Tensor:
    r"""
    Computes the mean `Hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`_, typically used for Support Vector
    Machines (SVMs). In the binary case it is defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    In the multi-class case, when ``multiclass_mode=None`` (default), ``multiclass_mode=MulticlassMode.CRAMMER_SINGER``
    or ``multiclass_mode="crammer-singer"``, this metric will compute the multi-class hinge loss defined by Crammer and
    Singer as:

    .. math::
        \text{Hinge loss} = \max\left(0, 1 - \hat{y}_y + \max_{i \ne y} (\hat{y}_i)\right)

    Where :math:`y \in {0, ..., \mathrm{C}}` is the target class (where :math:`\mathrm{C}` is the number of classes),
    and :math:`\hat{y} \in \mathbb{R}^\mathrm{C}` is the predicted output per class.

    In the multi-class case when ``multiclass_mode=MulticlassMode.ONE_VS_ALL`` or ``multiclass_mode='one-vs-all'``, this
    metric will use a one-vs-all approach to compute the hinge loss, giving a vector of C outputs where each entry pits
    that class against all remaining classes.

    This metric can optionally output the mean of the squared hinge loss by setting ``squared=True``

    Only accepts inputs with preds shape of (N) (binary) or (N, C) (multi-class) and target shape of (N).

    Args:
        preds: Predictions from model (as float outputs from decision function).
        target: Ground truth labels.
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss (default).
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.

    Raises:
        ValueError:
            If preds shape is not of size (N) or (N, C).
        ValueError:
            If target shape is not of size (N).
        ValueError:
            If ``multiclass_mode`` is not: None, ``MulticlassMode.CRAMMER_SINGER``, ``"crammer-singer"``,
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"``.

    Example (binary case):
        >>> import torch
        >>> from torchmetrics.functional import hinge
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> hinge(preds, target)
        tensor(0.3000)

    Example (default / multiclass case):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge(preds, target)
        tensor(2.9000)

    Example (multiclass example, one vs all mode):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge(preds, target, multiclass_mode="one-vs-all")
        tensor([2.2333, 1.5000, 1.2333])
    """
    measure, total = _hinge_update(preds, target, squared=squared, multiclass_mode=multiclass_mode)
    return _hinge_compute(measure, total)
