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
from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.functional.classification.hamming import _hamming_distance_compute, _hamming_distance_update
from torchmetrics.metric import Metric


class HammingDistance(Metric):
    r"""Computes the average `Hamming distance`_ (also known as Hamming loss) between targets and predictions:

    .. math::
        \text{Hamming distance} = \frac{1}{N \cdot L}\sum_i^N \sum_l^L 1(y_{il} \neq \hat{y_{il}})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label.

    Accepts all input types listed in :ref:`pages/classification:input types`.

    Args:
        threshold:
            Threshold for transforming probability or logit predictions to binary ``(0,1)`` predictions, in the case
            of binary or multi-label inputs. Default value of ``0.5`` corresponds to input being probabilities.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``threshold`` is not between ``0`` and ``1``.

    Example:
        >>> from torchmetrics import HammingDistance
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> hamming_distance = HammingDistance()
        >>> hamming_distance(preds, target)
        tensor(0.2500)

    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    correct: Tensor
    total: Tensor

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        See :ref:`pages/classification:input types` for more information on input types.

        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth labels
        """
        correct, total = _hamming_distance_update(preds, target, self.threshold)

        self.correct += correct
        self.total += total

    def compute(self) -> Tensor:
        """Computes hamming distance based on inputs passed in to ``update`` previously."""
        return _hamming_distance_compute(self.correct, self.total)
