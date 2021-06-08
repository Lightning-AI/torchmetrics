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
from typing import Any, Callable, Optional, Union

from torch import Tensor, tensor

from torchmetrics.functional.classification.hinge import MulticlassMode, _hinge_compute, _hinge_update
from torchmetrics.metric import Metric


class Hinge(Metric):
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
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss (default).
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.

    Raises:
        ValueError:
            If ``multiclass_mode`` is not: None, ``MulticlassMode.CRAMMER_SINGER``, ``"crammer-singer"``,
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"``.

    Example (binary case):
        >>> import torch
        >>> from torchmetrics import Hinge
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> hinge = Hinge()
        >>> hinge(preds, target)
        tensor(0.3000)

    Example (default / multiclass case):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = Hinge()
        >>> hinge(preds, target)
        tensor(2.9000)

    Example (multiclass example, one vs all mode):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = Hinge(multiclass_mode="one-vs-all")
        >>> hinge(preds, target)
        tensor([2.2333, 1.5000, 1.2333])
    """

    def __init__(
        self,
        squared: bool = False,
        multiclass_mode: Optional[Union[str, MulticlassMode]] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("measure", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

        if multiclass_mode not in (None, MulticlassMode.CRAMMER_SINGER, MulticlassMode.ONE_VS_ALL):
            raise ValueError(
                "The `multiclass_mode` should be either None / 'crammer-singer' / MulticlassMode.CRAMMER_SINGER"
                "(default) or 'one-vs-all' / MulticlassMode.ONE_VS_ALL,"
                f" got {multiclass_mode}."
            )

        self.squared = squared
        self.multiclass_mode = multiclass_mode

    def update(self, preds: Tensor, target: Tensor):
        measure, total = _hinge_update(preds, target, squared=self.squared, multiclass_mode=self.multiclass_mode)

        self.measure = measure + self.measure
        self.total = total + self.total

    def compute(self) -> Tensor:
        return _hinge_compute(self.measure, self.total)

    @property
    def is_differentiable(self) -> bool:
        return True
